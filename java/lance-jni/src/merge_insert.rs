// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::import_vec_to_rust;
use crate::traits::{FromJString, IntoJava};
use crate::{Error, JNIEnvExt, RT};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use jni::JNIEnv;
use jni::objects::{JByteArray, JObject, JString, JValueGen};
use jni::sys::{jbyteArray, jint, jlong};
use lance::dataset::scanner::ExprFilter;
use lance::dataset::{
    MergeInsertBuilder, MergeStats, SourceDedupeBehavior, WhenMatched, WhenNotMatched,
    WhenNotMatchedBySource,
};
use lance_core::datatypes::Schema;
use lance_index::mem_wal::MergedGeneration;
use prost::Message;
use std::mem::transmute;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMergeInsert<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,    // Dataset object
    jparam: JObject,      // MergeInsertParams object
    batch_address: jlong, // ArrowArrayStream address for source
) -> JObject<'a> {
    ok_or_throw!(
        env,
        inner_merge_insert(&mut env, jdataset, jparam, batch_address)
    )
}

/// Build a `MergeInsertJob` from the Java `MergeInsertParams`, sharing all the
/// param extraction between the committed and uncommitted merge entry points.
fn build_merge_job(
    env: &mut JNIEnv<'_>,
    jdataset: &JObject,
    jparam: &JObject,
) -> Result<lance::dataset::MergeInsertJob> {
    let on = extract_on(env, jparam)?;
    let when_matched = extract_when_matched(env, jparam)?;
    let when_not_matched = extract_when_not_matached(env, jparam)?;
    let when_not_matched_by_source_str = extract_when_not_matched_by_source_str(env, jparam)?;
    let when_not_matched_by_source_delete_expr =
        extract_when_not_matched_by_source_delete_expr(env, jparam)?;
    let conflict_retries = extract_conflict_retries(env, jparam)?;
    let retry_timeout_ms = extract_retry_timeout_ms(env, jparam)?;
    let skip_auto_cleanup = extract_skip_auto_cleanup(env, jparam)?;
    let use_index = extract_use_index(env, jparam)?;
    let source_dedupe_behavior = extract_source_dedupe_behavior(env, jparam)?;
    let marked_generations = extract_marked_generations(env, jparam)?;

    unsafe {
        let dataset = env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET)?;
        let when_not_matched_by_source = extract_when_not_matched_by_source(
            dataset.inner.schema(),
            when_not_matched_by_source_str.as_str(),
            when_not_matched_by_source_delete_expr,
        )?;
        let job = MergeInsertBuilder::try_new(Arc::new(dataset.clone().inner), on)?
            .when_matched(when_matched)
            .when_not_matched(when_not_matched)
            .when_not_matched_by_source(when_not_matched_by_source)
            .conflict_retries(conflict_retries)
            .retry_timeout(Duration::from_millis(retry_timeout_ms as u64))
            .skip_auto_cleanup(skip_auto_cleanup)
            .use_index(use_index)
            .source_dedupe_behavior(source_dedupe_behavior)
            .mark_generations_as_merged(marked_generations)
            .try_build()?;
        Ok(job)
    }
}

/// Uncommitted merge insert: runs the merge (writes new data fragments to
/// storage) but does NOT commit. Returns the resulting `Transaction` encoded as
/// protobuf bytes, so a distributed caller can ship it (e.g. Spark executor →
/// driver) and later combine + commit several such transactions as one.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMergeInsertUncommitted<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
    jparam: JObject,
    batch_address: jlong,
) -> jbyteArray {
    match inner_merge_insert_uncommitted(&mut env, jdataset, jparam, batch_address) {
        Ok(arr) => arr.into_raw(),
        Err(e) => {
            e.throw(&mut env);
            JByteArray::default().into_raw()
        }
    }
}

fn inner_merge_insert_uncommitted<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jparam: JObject,
    batch_address: jlong,
) -> Result<JByteArray<'local>> {
    let merge_insert_job = build_merge_job(env, &jdataset, &jparam)?;

    let uncommitted = unsafe {
        let stream_ptr = batch_address as *mut FFI_ArrowArrayStream;
        let source_stream = ArrowArrayStreamReader::from_raw(stream_ptr)?;
        RT.block_on(async move { merge_insert_job.execute_uncommitted(source_stream).await })?
    };

    let pb_txn = lance_table::format::pb::Transaction::from(&uncommitted.transaction);
    let bytes = pb_txn.encode_to_vec();
    let arr = env.new_byte_array(bytes.len() as jint)?;
    let i8_slice: &[i8] = unsafe { transmute(bytes.as_slice()) };
    env.set_byte_array_region(&arr, 0, i8_slice)?;
    Ok(arr)
}

/// Combine several uncommitted merge transactions (each protobuf-encoded
/// `Transaction` bytes from [`Java_org_lance_Dataset_nativeMergeInsertUncommitted`])
/// into a single transaction — unioning per-fragment deletion vectors — and
/// commit it as one operation. Returns the new committed `Dataset`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCommitMergeTransactions<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
    txn_bytes_array: JObject,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        inner_commit_merge_transactions(&mut env, jdataset, txn_bytes_array)
    )
}

fn inner_commit_merge_transactions<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    txn_bytes_array: JObject,
) -> Result<JObject<'local>> {
    use jni::objects::JObjectArray;
    use lance::dataset::{CommitBuilder, combine_merge_transactions};

    // Decode each byte[] element into a Rust Transaction.
    let array = JObjectArray::from(txn_bytes_array);
    let n = env.get_array_length(&array)?;
    let mut transactions = Vec::with_capacity(n as usize);
    for i in 0..n {
        let elem = env.get_object_array_element(&array, i)?;
        let bytes = env.convert_byte_array(JByteArray::from(elem))?;
        let pb_txn = lance_table::format::pb::Transaction::decode(bytes.as_slice())
            .map_err(|e| Error::input_error(format!("failed to decode transaction: {e}")))?;
        let txn = lance::dataset::transaction::Transaction::try_from(pb_txn)
            .map_err(|e| Error::input_error(format!("invalid transaction: {e}")))?;
        transactions.push(txn);
    }

    let new_ds = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&jdataset, NATIVE_DATASET)? };
        let base = Arc::new(dataset.clone().inner);
        RT.block_on(async move {
            let t_combine = std::time::Instant::now();
            let combined = combine_merge_transactions(base.as_ref(), transactions).await?;
            let combine_ms = t_combine.elapsed().as_millis();
            let t_commit = std::time::Instant::now();
            let out = CommitBuilder::new(base).execute(combined).await;
            eprintln!(
                "[jni-timing] combine={}ms commit={}ms",
                combine_ms,
                t_commit.elapsed().as_millis()
            );
            out
        })?
    };

    BlockingDataset { inner: new_ds }.into_java(env)
}

#[allow(clippy::too_many_arguments)]
fn inner_merge_insert<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jparam: JObject,
    batch_address: jlong,
) -> Result<JObject<'local>> {
    let merge_insert_job = build_merge_job(env, &jdataset, &jparam)?;

    let (new_ds, merge_stats) = unsafe {
        let stream_ptr = batch_address as *mut FFI_ArrowArrayStream;
        let source_stream = ArrowArrayStreamReader::from_raw(stream_ptr)?;
        RT.block_on(async move { merge_insert_job.execute_reader(source_stream).await })?
    };

    MergeResult(
        BlockingDataset {
            inner: Arc::try_unwrap(new_ds).unwrap(),
        },
        merge_stats,
    )
    .into_java(env)
}

fn extract_on<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<Vec<String>> {
    let on: JObject = env
        .call_method(jparam, "on", "()Ljava/util/List;", &[])?
        .l()?;
    env.get_strings(&on)
}

fn extract_when_matched<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<WhenMatched> {
    let when_matched: JString = env
        .call_method(jparam, "whenMatchedValue", "()Ljava/lang/String;", &[])?
        .l()?
        .into();
    let when_matched = when_matched.extract(env)?;

    let when_matched_update_expr = env
        .call_method(
            jparam,
            "whenMatchedUpdateExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let when_matched_update_expr = env.get_string_opt(&when_matched_update_expr)?;

    match when_matched.as_str() {
        "UpdateAll" => Ok(WhenMatched::UpdateAll),
        "DoNothing" => Ok(WhenMatched::DoNothing),
        "UpdateIf" => match when_matched_update_expr {
            Some(expr) => Ok(WhenMatched::UpdateIf(expr)),
            None => Err(Error::input_error("No matched updated expr".to_string())),
        },
        "Fail" => Ok(WhenMatched::Fail),
        "Delete" => Ok(WhenMatched::Delete),
        _ => Err(Error::input_error(format!(
            "Illegal when_matched: {when_matched}",
        ))),
    }
}

fn extract_when_not_matached<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<WhenNotMatched> {
    let when_not_matched: JString = env
        .call_method(jparam, "whenNotMatchedValue", "()Ljava/lang/String;", &[])?
        .l()?
        .into();
    let when_not_matched = when_not_matched.extract(env)?;

    match when_not_matched.as_str() {
        "InsertAll" => Ok(WhenNotMatched::InsertAll),
        "DoNothing" => Ok(WhenNotMatched::DoNothing),
        _ => Err(Error::input_error(format!(
            "Illegal when_not_matched: {when_not_matched}",
        ))),
    }
}

fn extract_when_not_matched_by_source_str<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<String> {
    let when_not_matched_by_source: JString = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceValue",
            "()Ljava/lang/String;",
            &[],
        )?
        .l()?
        .into();
    when_not_matched_by_source.extract(env)
}

fn extract_when_not_matched_by_source_delete_expr<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<Option<ExprFilter>> {
    let when_not_matched_by_source_delete_expr = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceDeleteExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    if let Some(expr) = env.get_string_opt(&when_not_matched_by_source_delete_expr)? {
        return Ok(Some(ExprFilter::Sql(expr)));
    }

    let when_not_matched_by_source_delete_substrait_expr = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceDeleteSubstraitExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    match env.get_bytes_opt(&when_not_matched_by_source_delete_substrait_expr)? {
        Some(expr) => Ok(Some(ExprFilter::Substrait(expr.to_vec()))),
        None => Ok(None),
    }
}

fn extract_when_not_matched_by_source(
    schema: &Schema,
    when_not_matched_by_source: &str,
    when_not_matched_by_source_delete_expr: Option<ExprFilter>,
) -> Result<WhenNotMatchedBySource> {
    match when_not_matched_by_source {
        "Keep" => Ok(WhenNotMatchedBySource::Keep),
        "Delete" => Ok(WhenNotMatchedBySource::Delete),
        "DeleteIf" => match when_not_matched_by_source_delete_expr {
            Some(expr) => Ok(WhenNotMatchedBySource::DeleteIf(
                expr.to_datafusion(schema, schema)?,
            )),
            None => Err(Error::input_error(format!(
                "No delete expr when not matched by source is: {when_not_matched_by_source}",
            ))),
        },
        _ => Err(Error::input_error(format!(
            "Illegal when_not_matched_by_source: {when_not_matched_by_source}",
        ))),
    }
}

fn extract_conflict_retries<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<u32> {
    let retries = env
        .call_method(jparam, "conflictRetries", "()I", &[])?
        .i()? as u32;
    Ok(retries)
}

fn extract_retry_timeout_ms<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<u64> {
    let timeout_ms = env.call_method(jparam, "retryTimeoutMs", "()J", &[])?.j()? as u64;
    Ok(timeout_ms)
}

fn extract_skip_auto_cleanup<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<bool> {
    let skip_auto_cleanup = env
        .call_method(jparam, "skipAutoCleanup", "()Z", &[])?
        .z()?;
    Ok(skip_auto_cleanup)
}

fn extract_use_index<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<bool> {
    let use_index = env.call_method(jparam, "useIndex", "()Z", &[])?.z()?;
    Ok(use_index)
}

fn extract_source_dedupe_behavior<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<SourceDedupeBehavior> {
    let behavior: JString = env
        .call_method(
            jparam,
            "sourceDedupeBehaviorValue",
            "()Ljava/lang/String;",
            &[],
        )?
        .l()?
        .into();
    let behavior = behavior.extract(env)?;
    match behavior.as_str() {
        "Fail" => Ok(SourceDedupeBehavior::Fail),
        "FirstSeen" => Ok(SourceDedupeBehavior::FirstSeen),
        _ => Err(Error::input_error(format!(
            "Illegal source_dedupe_behavior: {behavior}",
        ))),
    }
}

fn extract_marked_generations<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<Vec<MergedGeneration>> {
    let list = env
        .call_method(jparam, "markedGenerations", "()Ljava/util/List;", &[])?
        .l()?;
    import_vec_to_rust(env, &list, |env, obj| {
        let shard_id: JString = env
            .call_method(&obj, "shardId", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let shard_id = shard_id.extract(env)?;
        let generation = env.call_method(&obj, "generation", "()J", &[])?.j()? as u64;
        let uuid = Uuid::parse_str(&shard_id)
            .map_err(|e| Error::input_error(format!("Invalid shard_id UUID: {}", e)))?;
        Ok(MergedGeneration::new(uuid, generation))
    })
}

const MERGE_STATS_CLASS: &str = "org/lance/merge/MergeInsertStats";
const MERGE_STATS_CONSTRUCTOR_SIG: &str = "(JJJIJJ)V";
const MERGE_RESULT_CLASS: &str = "org/lance/merge/MergeInsertResult";
const MERGE_RESULT_CONSTRUCTOR_SIG: &str =
    "(Lorg/lance/Dataset;Lorg/lance/merge/MergeInsertStats;)V";

impl IntoJava for MergeStats {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object(
            MERGE_STATS_CLASS,
            MERGE_STATS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.num_inserted_rows as i64),
                JValueGen::Long(self.num_updated_rows as i64),
                JValueGen::Long(self.num_deleted_rows as i64),
                JValueGen::Int(self.num_attempts as i32),
                JValueGen::Long(self.bytes_written as i64),
                JValueGen::Long(self.num_files_written as i64),
            ],
        )?)
    }
}

struct MergeResult(BlockingDataset, MergeStats);

impl IntoJava for MergeResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let jdataset = self.0.into_java(env)?;
        let jstats = self.1.into_java(env)?;
        Ok(env.new_object(
            MERGE_RESULT_CLASS,
            MERGE_RESULT_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&jdataset), JValueGen::Object(&jstats)],
        )?)
    }
}
