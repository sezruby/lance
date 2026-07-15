// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JNI bindings for [`lance::index::vector::external::ExternalIvfPqIndex`].
//!
//! Java surface (corresponds to `org.lance.index.external.ExternalIvfPqIndex`):
//!
//!   - `nativeBuild(...) -> String` (returns the index UUID directory name)
//!   - `nativeOpen(uri) -> long` (opaque handle)
//!   - `nativeClose(handle)`
//!   - `nativeSearch(handle, query, k, nprobes, refineFactor, deletedBitmap) -> SearchResult[]`
//!   - `nativeFetchRows(handle, filePaths, rowIndices, projection) -> Arrow IPC bytes`
//!
//! `SearchResult[]` is returned via Java object construction. `nativeFetchRows`
//! returns Arrow IPC stream bytes so the caller can decode with their preferred
//! Arrow Java reader without us needing to bridge `RecordBatch` directly.
//!
//! The RowFilter API is exposed as an optional `byte[] deletedBitmap` (LE
//! little-endian Roaring bitmap). Rows whose `(file_id << 32) | row_index` is
//! present are dropped during refinement. This sidesteps cross-language
//! callbacks for Phase 1; the trait-based `RowFilter` stays available for Rust
//! callers.

use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JObjectArray, JString, JValue};
use jni::sys::{jbyteArray, jfloat, jint, jlong, jlongArray};
use lance::index::vector::external::{
    ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetFileSpec, ParquetRowKey, RerankStore,
    RowFilter, SearchResult,
};
use lance_linalg::distance::MetricType;

use crate::error::{Error, Result};
use crate::traits::FromJString;
use crate::RT;

/// RowFilter implementation that holds a sorted list of deleted rids. Caller
/// passes the rids as a packed `(file_id << 32) | row_index` u64 array. We
/// resolve the file_id via the index manifest at search time.
struct DeletedRidFilter {
    /// Sorted, deduped deleted rids. Binary search per refinement candidate.
    deleted: Vec<u64>,
    /// Path → file_id index for fast lookup.
    file_id_by_path: std::collections::HashMap<String, u32>,
}

impl RowFilter for DeletedRidFilter {
    fn keep(&self, file_path: &str, row_index: u64) -> bool {
        let Some(&file_id) = self.file_id_by_path.get(file_path) else {
            // unknown file → keep (search would have errored before refinement
            // on an unknown file anyway)
            return true;
        };
        let rid = ((file_id as u64) << 32) | row_index;
        self.deleted.binary_search(&rid).is_err()
    }
}

/// Build a non-empty filter from a Java `byte[]` of u64-LE deleted rids.
/// Returns `Ok(None)` when the byte array is null or empty.
fn build_filter_from_bytes(
    env: &mut JNIEnv,
    deleted_bytes: &JByteArray,
    file_id_by_path: std::collections::HashMap<String, u32>,
) -> Result<Option<DeletedRidFilter>> {
    if deleted_bytes.is_null() {
        return Ok(None);
    }
    let bytes = env.convert_byte_array(deleted_bytes)?;
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.len() % 8 != 0 {
        return Err(Error::input_error(format!(
            "deletedBitmap byte length {} is not a multiple of 8 (u64 LE)",
            bytes.len()
        )));
    }
    let mut deleted: Vec<u64> = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let arr: [u8; 8] = chunk.try_into().expect("chunks_exact 8");
        deleted.push(u64::from_le_bytes(arr));
    }
    deleted.sort_unstable();
    deleted.dedup();
    Ok(Some(DeletedRidFilter {
        deleted,
        file_id_by_path,
    }))
}

fn parse_rerank_store(s: &str) -> Result<RerankStore> {
    match s.to_ascii_lowercase().as_str() {
        "none" | "" => Ok(RerankStore::None),
        "sq8" => Ok(RerankStore::Sq8),
        "flat" => Ok(RerankStore::Flat),
        other => Err(Error::input_error(format!(
            "unsupported rerank store '{other}'; expected one of None, Sq8, Flat"
        ))),
    }
}

fn parse_metric(s: &str) -> Result<MetricType> {
    match s.to_ascii_lowercase().as_str() {
        "l2" => Ok(MetricType::L2),
        "cosine" => Ok(MetricType::Cosine),
        "dot" => Ok(MetricType::Dot),
        other => Err(Error::input_error(format!(
            "unsupported metric '{other}'; expected one of L2, Cosine, Dot"
        ))),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeBuild<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    file_paths: JObjectArray<'local>,
    vector_column: JString<'local>,
    output_uri: JString<'local>,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString<'local>,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_build(
            &mut env,
            file_paths,
            vector_column,
            output_uri,
            num_partitions,
            num_sub_vectors,
            num_bits_per_sub_vector,
            metric,
            max_iters,
            sample_rate,
            seed,
            rerank_store,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_build<'local>(
    env: &mut JNIEnv<'local>,
    file_paths: JObjectArray<'local>,
    vector_column: JString<'local>,
    output_uri: JString<'local>,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString<'local>,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString<'local>,
) -> Result<JObject<'local>> {
    let n = env.get_array_length(&file_paths)?;
    let mut files: Vec<ParquetFileSpec> = Vec::with_capacity(n as usize);
    for i in 0..n {
        let elem: JString = env.get_object_array_element(&file_paths, i)?.into();
        let path: String = elem.extract(env)?;
        files.push(ParquetFileSpec::of(path));
    }

    let vector_column_str: String = vector_column.extract(env)?;
    let output_uri_str: String = output_uri.extract(env)?;
    let metric_str: String = metric.extract(env)?;
    let rerank_store_str: String = rerank_store.extract(env)?;

    let params = ExternalIvfPqIndexParams::builder()
        .num_partitions(num_partitions as usize)
        .num_sub_vectors(num_sub_vectors as usize)
        .num_bits_per_sub_vector(num_bits_per_sub_vector as usize)
        .metric(parse_metric(&metric_str)?)
        .max_iters(max_iters as usize)
        .sample_rate(sample_rate as usize)
        .seed(seed as u64)
        .rerank_store(parse_rerank_store(&rerank_store_str)?)
        .build();

    let uuid = RT.block_on(async move {
        ExternalIvfPqIndex::build(files, &vector_column_str, &output_uri_str, params).await
    })?;

    let uuid_str = uuid.to_string();
    let j = env.new_string(&uuid_str)?;
    Ok(j.into())
}

// ---- distributed build --------------------------------------------------------

/// Build `ExternalIvfPqIndexParams` from the flat JNI scalar args (shared by the
/// single-box and distributed build entry points).
#[allow(clippy::too_many_arguments)]
fn params_from_args(
    env: &mut JNIEnv,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString,
) -> Result<ExternalIvfPqIndexParams> {
    let metric_str: String = metric.extract(env)?;
    let rerank_store_str: String = rerank_store.extract(env)?;
    Ok(ExternalIvfPqIndexParams::builder()
        .num_partitions(num_partitions as usize)
        .num_sub_vectors(num_sub_vectors as usize)
        .num_bits_per_sub_vector(num_bits_per_sub_vector as usize)
        .metric(parse_metric(&metric_str)?)
        .max_iters(max_iters as usize)
        .sample_rate(sample_rate as usize)
        .seed(seed as u64)
        .rerank_store(parse_rerank_store(&rerank_store_str)?)
        .build())
}

fn extract_string_array(env: &mut JNIEnv, arr: &JObjectArray) -> Result<Vec<String>> {
    let n = env.get_array_length(arr)?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let s: JString = env.get_object_array_element(arr, i)?.into();
        out.push(s.extract(env)?);
    }
    Ok(out)
}

/// Driver phase 1: train quantizers on a sample and return the broadcast payload
/// as a `byte[]` (see `BroadcastPayload::to_bytes`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeTrainBroadcast<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    file_paths: JObjectArray<'local>,
    vector_column: JString<'local>,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString<'local>,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString<'local>,
) -> jbyteArray {
    let result = (|| -> Result<Vec<u8>> {
        let paths = extract_string_array(&mut env, &file_paths)?;
        let files: Vec<ParquetFileSpec> = paths.into_iter().map(ParquetFileSpec::of).collect();
        let vector_column_str: String = vector_column.extract(&mut env)?;
        let params = params_from_args(
            &mut env,
            num_partitions,
            num_sub_vectors,
            num_bits_per_sub_vector,
            metric,
            max_iters,
            sample_rate,
            seed,
            rerank_store,
        )?;
        let payload = RT.block_on(async move {
            lance::index::vector::external::train_broadcast_payload(
                files,
                &vector_column_str,
                &params,
            )
            .await
        })?;
        Ok(payload.to_bytes())
    })();
    match result {
        Ok(bytes) => env
            .byte_array_from_slice(&bytes)
            .map(|a| a.into_raw())
            .unwrap_or(std::ptr::null_mut()),
        Err(e) => {
            let _ = env.throw(("java/io/IOException", format!("{e}")));
            std::ptr::null_mut()
        }
    }
}

/// Executor phase 2: build one file shard's partition output to `shard_uri` using
/// the broadcast payload. `file_id_offset` is the global position of this shard's
/// first file in the full sorted file list. Returns the sidecar shard's row count
/// (0 when no rerank store) so the driver can prefix-sum ordinal bases; the sidecar
/// itself is written to `<shard_uri>.rerank`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeBuildShard<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    payload_bytes: JByteArray<'local>,
    file_paths: JObjectArray<'local>,
    vector_column: JString<'local>,
    file_id_offset: jint,
    shard_uri: JString<'local>,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString<'local>,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString<'local>,
) -> jlong {
    let result = (|| -> Result<i64> {
        let payload = payload_from_jbytes(&mut env, &payload_bytes)?;
        let paths = extract_string_array(&mut env, &file_paths)?;
        let files: Vec<ParquetFileSpec> = paths.into_iter().map(ParquetFileSpec::of).collect();
        let vector_column_str: String = vector_column.extract(&mut env)?;
        let shard_uri_str: String = shard_uri.extract(&mut env)?;
        let params = params_from_args(
            &mut env,
            num_partitions,
            num_sub_vectors,
            num_bits_per_sub_vector,
            metric,
            max_iters,
            sample_rate,
            seed,
            rerank_store,
        )?;
        let res = RT.block_on(async move {
            lance::index::vector::external::build_shard_to_parquet(
                &payload,
                files,
                &vector_column_str,
                file_id_offset as u32,
                &params,
                &shard_uri_str,
            )
            .await
        })?;
        Ok(res.sidecar.map(|s| s.rows as i64).unwrap_or(0))
    })();
    match result {
        Ok(rows) => rows,
        Err(e) => {
            let _ = env.throw(("java/io/IOException", format!("{e}")));
            0
        }
    }
}

/// Driver phase 3: merge shard parquets into `<index_dir_uri>/index.idx` and write
/// `manifest.json` alongside, leaving an openable index directory. `file_paths` is
/// the full sorted file list (position = global file_id, matching the shard rids).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeMergeShards<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    payload_bytes: JByteArray<'local>,
    shard_uris: JObjectArray<'local>,
    file_paths: JObjectArray<'local>,
    vector_column: JString<'local>,
    index_dir_uri: JString<'local>,
    num_partitions: jint,
    num_sub_vectors: jint,
    num_bits_per_sub_vector: jint,
    metric: JString<'local>,
    max_iters: jint,
    sample_rate: jint,
    seed: jlong,
    rerank_store: JString<'local>,
    // Sidecar shards in global file order (parallel arrays); empty when no store.
    sidecar_uris: JObjectArray<'local>,
    sidecar_rows: jlongArray,
) {
    ok_or_throw_without_return!(env, {
        (|| -> Result<()> {
            let payload = payload_from_jbytes(&mut env, &payload_bytes)?;
            let uris = extract_string_array(&mut env, &shard_uris)?;
            let paths = extract_string_array(&mut env, &file_paths)?;
            let vector_column_str: String = vector_column.extract(&mut env)?;
            let index_dir_str: String = index_dir_uri.extract(&mut env)?;
            let params = params_from_args(
                &mut env,
                num_partitions,
                num_sub_vectors,
                num_bits_per_sub_vector,
                metric,
                max_iters,
                sample_rate,
                seed,
                rerank_store,
            )?;
            // Zip sidecar (uri, rows) parallel arrays.
            let sc_uris = extract_string_array(&mut env, &sidecar_uris)?;
            let sidecar_shards: Vec<(String, u64)> = if sc_uris.is_empty() {
                Vec::new()
            } else {
                let arr = unsafe { jni::objects::JLongArray::from_raw(sidecar_rows) };
                let len = env.get_array_length(&arr)? as usize;
                let mut rows = vec![0i64; len];
                env.get_long_array_region(&arr, 0, &mut rows)?;
                sc_uris
                    .into_iter()
                    .zip(rows.into_iter())
                    .map(|(u, r)| (u, r as u64))
                    .collect()
            };
            RT.block_on(async move {
                lance::index::vector::external::merge_shards_to_index(
                    &payload,
                    &uris,
                    &paths,
                    &vector_column_str,
                    &index_dir_str,
                    &params,
                    &sidecar_shards,
                )
                .await
            })?;
            Ok(())
        })()
    });
}

fn payload_from_jbytes(
    env: &mut JNIEnv,
    bytes: &JByteArray,
) -> Result<lance::index::vector::external::BroadcastPayload> {
    let raw = env.convert_byte_array(bytes)?;
    Ok(lance::index::vector::external::BroadcastPayload::from_bytes(&raw)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeOpen<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    uri: JString<'local>,
) -> jlong {
    ok_or_throw_with_return!(env, inner_open(&mut env, uri), 0i64)
}

fn inner_open(env: &mut JNIEnv, uri: JString) -> Result<jlong> {
    let uri_str: String = uri.extract(env)?;
    let idx = RT.block_on(async move { ExternalIvfPqIndex::open(&uri_str).await })?;
    let boxed = Box::new(Arc::new(idx));
    Ok(Box::into_raw(boxed) as jlong)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeClose(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: handle came from a previous Box::into_raw on an Arc<ExternalIvfPqIndex>;
    // taking it back into a Box drops the Arc.
    unsafe {
        let _ = Box::from_raw(handle as *mut Arc<ExternalIvfPqIndex>);
    }
}

fn handle_to_idx(handle: jlong) -> Result<Arc<ExternalIvfPqIndex>> {
    if handle == 0 {
        return Err(Error::input_error(
            "ExternalIvfPqIndex handle is null (closed?)".to_string(),
        ));
    }
    // SAFETY: handle is a non-null pointer from Box::into_raw; cloning the Arc
    // inside doesn't take ownership.
    let arc_ref = unsafe { &*(handle as *const Arc<ExternalIvfPqIndex>) };
    Ok(arc_ref.clone())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeSearch<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    query: jni::objects::JFloatArray<'local>,
    k: jint,
    nprobes: jint,
    refine_factor: jint,
    deleted_bitmap: JByteArray<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_search(
            &mut env,
            handle,
            query,
            k,
            nprobes,
            refine_factor,
            deleted_bitmap,
        )
    )
}

fn inner_search<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    query: jni::objects::JFloatArray<'local>,
    k: jint,
    nprobes: jint,
    refine_factor: jint,
    deleted_bitmap: JByteArray<'local>,
) -> Result<JObject<'local>> {
    let idx = handle_to_idx(handle)?;
    let q_len = env.get_array_length(&query)?;
    let mut q_buf: Vec<jfloat> = vec![0.0; q_len as usize];
    env.get_float_array_region(&query, 0, &mut q_buf)?;

    // Build path → file_id map for the optional filter.
    let mut file_id_by_path: std::collections::HashMap<String, u32> =
        std::collections::HashMap::with_capacity(idx.num_files());
    for fid in 0..idx.num_files() as u32 {
        if let Some(p) = idx.file_path(fid) {
            file_id_by_path.insert(p.to_string(), fid);
        }
    }
    let filter = build_filter_from_bytes(env, &deleted_bitmap, file_id_by_path)?;

    let results: Vec<SearchResult> = RT.block_on(async {
        idx.search(
            &q_buf,
            k as usize,
            nprobes as usize,
            refine_factor as usize,
            filter.as_ref().map(|f| f as &dyn RowFilter),
        )
        .await
    })?;

    // Build SearchResult[] in Java.
    let result_class = env.find_class("org/lance/index/external/SearchResult")?;
    let array = env.new_object_array(results.len() as i32, &result_class, JObject::null())?;
    for (i, r) in results.iter().enumerate() {
        let path = env.new_string(&r.file_path)?;
        let obj = env.new_object(
            &result_class,
            "(Ljava/lang/String;JF)V",
            &[
                JValue::Object(&path),
                JValue::Long(r.row_index as i64),
                JValue::Float(r.distance),
            ],
        )?;
        env.set_object_array_element(&array, i as i32, obj)?;
    }
    Ok(array.into())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeSearchBatch<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    queries: JObjectArray<'local>,
    k: jint,
    nprobes: jint,
    refine_factor: jint,
    deleted_bitmap: JByteArray<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_search_batch(
            &mut env,
            handle,
            queries,
            k,
            nprobes,
            refine_factor,
            deleted_bitmap,
        )
    )
}

fn inner_search_batch<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    queries: JObjectArray<'local>,
    k: jint,
    nprobes: jint,
    refine_factor: jint,
    deleted_bitmap: JByteArray<'local>,
) -> Result<JObject<'local>> {
    let idx = handle_to_idx(handle)?;

    // Decode float[][] queries → Vec<Vec<f32>> (owned).
    let n_queries = env.get_array_length(&queries)?;
    let mut owned: Vec<Vec<jfloat>> = Vec::with_capacity(n_queries as usize);
    for i in 0..n_queries {
        let row_obj = env.get_object_array_element(&queries, i)?;
        let row: jni::objects::JFloatArray = row_obj.into();
        let row_len = env.get_array_length(&row)?;
        let mut buf: Vec<jfloat> = vec![0.0; row_len as usize];
        env.get_float_array_region(&row, 0, &mut buf)?;
        owned.push(buf);
    }
    let query_refs: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();

    let mut file_id_by_path: std::collections::HashMap<String, u32> =
        std::collections::HashMap::with_capacity(idx.num_files());
    for fid in 0..idx.num_files() as u32 {
        if let Some(p) = idx.file_path(fid) {
            file_id_by_path.insert(p.to_string(), fid);
        }
    }
    let filter = build_filter_from_bytes(env, &deleted_bitmap, file_id_by_path)?;

    let batched: Vec<Vec<SearchResult>> = RT.block_on(async {
        idx.search_batch(
            &query_refs,
            k as usize,
            nprobes as usize,
            refine_factor as usize,
            filter.as_ref().map(|f| f as &dyn RowFilter),
        )
        .await
    })?;

    // Build SearchResult[][] in Java.
    let result_class = env.find_class("org/lance/index/external/SearchResult")?;
    let result_array_class = env.find_class("[Lorg/lance/index/external/SearchResult;")?;
    let outer = env.new_object_array(batched.len() as i32, &result_array_class, JObject::null())?;
    for (qi, results) in batched.iter().enumerate() {
        let inner =
            env.new_object_array(results.len() as i32, &result_class, JObject::null())?;
        for (i, r) in results.iter().enumerate() {
            let path = env.new_string(&r.file_path)?;
            let obj = env.new_object(
                &result_class,
                "(Ljava/lang/String;JF)V",
                &[
                    JValue::Object(&path),
                    JValue::Long(r.row_index as i64),
                    JValue::Float(r.distance),
                ],
            )?;
            env.set_object_array_element(&inner, i as i32, obj)?;
        }
        env.set_object_array_element(&outer, qi as i32, inner)?;
    }
    Ok(outer.into())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeFetchRows<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    file_paths: JObjectArray<'local>,
    row_indices: jlongArray,
    projection: JObjectArray<'local>,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_fetch_rows(&mut env, handle, file_paths, row_indices, projection),
        std::ptr::null_mut::<jni::sys::_jobject>() as jbyteArray
    )
}

fn inner_fetch_rows<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    file_paths: JObjectArray<'local>,
    row_indices: jlongArray,
    projection: JObjectArray<'local>,
) -> Result<jbyteArray> {
    let idx = handle_to_idx(handle)?;

    // Decode file_paths
    let n_paths = env.get_array_length(&file_paths)?;
    let mut paths: Vec<String> = Vec::with_capacity(n_paths as usize);
    for i in 0..n_paths {
        let elem: JString = env.get_object_array_element(&file_paths, i)?.into();
        let path: String = elem.extract(env)?;
        paths.push(path);
    }

    // Decode row_indices
    let row_indices_obj = unsafe { jni::objects::JLongArray::from_raw(row_indices) };
    let n_rows = env.get_array_length(&row_indices_obj)?;
    if n_rows != n_paths {
        return Err(Error::input_error(format!(
            "fetchRows: row_indices length {n_rows} != file_paths length {n_paths}"
        )));
    }
    let mut rids_i64: Vec<jlong> = vec![0; n_rows as usize];
    env.get_long_array_region(&row_indices_obj, 0, &mut rids_i64)?;

    let row_keys: Vec<ParquetRowKey> = paths
        .into_iter()
        .zip(rids_i64.iter().map(|&v| v as u64))
        .map(|(p, r)| ParquetRowKey::of(p, r))
        .collect();

    // Decode projection
    let n_proj = env.get_array_length(&projection)?;
    let mut proj_strings: Vec<String> = Vec::with_capacity(n_proj as usize);
    for i in 0..n_proj {
        let elem: JString = env.get_object_array_element(&projection, i)?.into();
        let s: String = elem.extract(env)?;
        proj_strings.push(s);
    }
    let proj_refs: Vec<&str> = proj_strings.iter().map(|s| s.as_str()).collect();

    // Run fetch
    let batch = RT.block_on(async { idx.fetch_rows(&row_keys, &proj_refs).await })?;

    // Serialize to Arrow IPC stream bytes.
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| Error::io_error(format!("ipc writer init: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::io_error(format!("ipc write: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::io_error(format!("ipc finish: {e}")))?;
    }
    let jbyte_slice: &[jni::sys::jbyte] = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const jni::sys::jbyte, buf.len())
    };
    let array = env.new_byte_array(buf.len() as i32)?;
    env.set_byte_array_region(&array, 0, jbyte_slice)?;
    Ok(array.into_raw())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeNumPartitions(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    let idx = match handle_to_idx(handle) {
        Ok(i) => i,
        Err(_) => return -1,
    };
    idx.num_partitions() as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeNumFiles(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    let idx = match handle_to_idx(handle) {
        Ok(i) => i,
        Err(_) => return -1,
    };
    idx.num_files() as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalIvfPqIndex_nativeVectorColumn<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
) -> JObject<'local> {
    let idx = match handle_to_idx(handle) {
        Ok(i) => i,
        Err(e) => {
            e.throw(&mut env);
            return JObject::null();
        }
    };
    match env.new_string(idx.vector_column()) {
        Ok(s) => s.into(),
        Err(e) => {
            Error::runtime_error(format!("new_string: {e}")).throw(&mut env);
            JObject::null()
        }
    }
}
