// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JNI bindings for [`lance::index::scalar_external::ExternalBtreeIndex`].
//!
//! The scalar analog of [`crate::external_index`] (the external *vector* index).
//! Java surface (corresponds to `org.lance.index.external.ExternalBtreeIndex`):
//!
//!   - `nativeBuild(filePaths, keyColumn, outputUri, batchSize) -> String` (UUID)
//!   - `nativeOpen(uri) -> long` (opaque handle), `nativeClose(handle)`
//!   - `nativeSearchKeys(handle, keyType, longKeys, stringKeys, deletedRids) -> SearchResult[]`
//!   - `nativeFetchRows(handle, filePaths, rowIndices, projection) -> Arrow IPC bytes`
//!   - `nativeNumFiles(handle) -> int`, `nativeKeyColumn(handle) -> String`
//!
//! The deleted-rids filter (packed `(file_id << 32) | row` u64-LE), the Arrow-IPC
//! `fetchRows` serialization, and the opaque `Box<Arc<...>>` handle scheme are all
//! shared with, or mirror, the vector JNI in [`crate::external_index`]. Only the
//! query surface differs: a scalar `IsIn` over `ScalarValue` keys returning the
//! matched-target-identity SET, versus the vector index's top-k.

use std::sync::Arc;

use datafusion_common::ScalarValue;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JLongArray, JObject, JObjectArray, JString, JValue};
use jni::sys::{jbyteArray, jint, jlong, jlongArray};
use lance::index::scalar_external::{
    ExternalBtreeIndex, ExternalBtreeIndexParams, ParquetFileSpec, ParquetRowKey, RowFilter,
    SearchResult,
};

use crate::RT;
use crate::error::{Error, Result};
use crate::external_index::{build_filter_from_bytes, record_batch_to_ipc_jbytes};
use crate::traits::FromJString;

// ---- build / open / close -----------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeBuild<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    file_paths: JObjectArray<'local>,
    key_column: JString<'local>,
    output_uri: JString<'local>,
    batch_size: jlong,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_build(&mut env, file_paths, key_column, output_uri, batch_size)
    )
}

fn inner_build<'local>(
    env: &mut JNIEnv<'local>,
    file_paths: JObjectArray<'local>,
    key_column: JString<'local>,
    output_uri: JString<'local>,
    batch_size: jlong,
) -> Result<JObject<'local>> {
    let n = env.get_array_length(&file_paths)?;
    let mut files: Vec<ParquetFileSpec> = Vec::with_capacity(n as usize);
    for i in 0..n {
        let elem: JString = env.get_object_array_element(&file_paths, i)?.into();
        let path: String = elem.extract(env)?;
        files.push(ParquetFileSpec::of(path));
    }

    let key_column_str: String = key_column.extract(env)?;
    let output_uri_str: String = output_uri.extract(env)?;
    if batch_size <= 0 {
        return Err(Error::input_error(format!(
            "batchSize must be positive, got {batch_size}"
        )));
    }
    let params = ExternalBtreeIndexParams::new().with_batch_size(batch_size as u64);

    let uuid = RT.block_on(async move {
        ExternalBtreeIndex::build(files, &key_column_str, &output_uri_str, params).await
    })?;

    let j = env.new_string(uuid.to_string())?;
    Ok(j.into())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeOpen<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    uri: JString<'local>,
) -> jlong {
    ok_or_throw_with_return!(env, inner_open(&mut env, uri), 0i64)
}

fn inner_open(env: &mut JNIEnv, uri: JString) -> Result<jlong> {
    let uri_str: String = uri.extract(env)?;
    let idx = RT.block_on(async move { ExternalBtreeIndex::open(&uri_str).await })?;
    let boxed = Box::new(Arc::new(idx));
    Ok(Box::into_raw(boxed) as jlong)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeClose(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: handle came from a previous Box::into_raw on an Arc<ExternalBtreeIndex>;
    // taking it back into a Box drops the Arc.
    unsafe {
        let _ = Box::from_raw(handle as *mut Arc<ExternalBtreeIndex>);
    }
}

fn handle_to_idx(handle: jlong) -> Result<Arc<ExternalBtreeIndex>> {
    if handle == 0 {
        return Err(Error::input_error(
            "ExternalBtreeIndex handle is null (closed?)".to_string(),
        ));
    }
    // SAFETY: handle is a non-null pointer from Box::into_raw; cloning the Arc
    // inside doesn't take ownership.
    let arc_ref = unsafe { &*(handle as *const Arc<ExternalBtreeIndex>) };
    Ok(arc_ref.clone())
}

// ---- search --------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeSearchKeys<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    key_type: JString<'local>,
    long_keys: jlongArray,
    string_keys: JObjectArray<'local>,
    deleted_rids: JByteArray<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_search_keys(
            &mut env,
            handle,
            key_type,
            long_keys,
            string_keys,
            deleted_rids,
        )
    )
}

fn inner_search_keys<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    key_type: JString<'local>,
    long_keys: jlongArray,
    string_keys: JObjectArray<'local>,
    deleted_rids: JByteArray<'local>,
) -> Result<JObject<'local>> {
    let idx = handle_to_idx(handle)?;

    let key_type_str: String = key_type.extract(env)?;
    let long_keys_arr = unsafe { JLongArray::from_raw(long_keys) };
    let keys = build_scalar_keys(env, &key_type_str, &long_keys_arr, &string_keys)?;

    // Build path -> file_id map for the optional deleted-rids filter, identical to
    // the vector index's search path.
    let mut file_id_by_path: std::collections::HashMap<String, u32> =
        std::collections::HashMap::with_capacity(idx.num_files());
    for fid in 0..idx.num_files() as u32 {
        if let Some(p) = idx.file_path(fid) {
            file_id_by_path.insert(p.to_string(), fid);
        }
    }
    let filter = build_filter_from_bytes(env, &deleted_rids, file_id_by_path)?;

    let results: Vec<SearchResult> = RT.block_on(async {
        idx.search_keys(&keys, filter.as_ref().map(|f| f as &dyn RowFilter))
            .await
    })?;

    // Build SearchResult[] in Java (reuses the vector index's SearchResult class).
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

/// Decode the caller's keys into `ScalarValue`s of the type named by `key_type`.
/// Only the array matching `key_type` is read; the other is ignored (Java passes
/// an empty array for it). Supports `int64` / `int32` (from `long_keys`) and
/// `utf8` (from `string_keys`) — the common Delta / Iceberg merge-key types. The
/// `ScalarValue` type must match the index's key-column Arrow type for `IsIn` to
/// match rows.
fn build_scalar_keys(
    env: &mut JNIEnv,
    key_type: &str,
    long_keys: &JLongArray,
    string_keys: &JObjectArray,
) -> Result<Vec<ScalarValue>> {
    match key_type.to_ascii_lowercase().as_str() {
        "int64" => {
            let longs = read_long_array(env, long_keys)?;
            Ok(longs
                .into_iter()
                .map(|k| ScalarValue::Int64(Some(k)))
                .collect())
        }
        "int32" => {
            let longs = read_long_array(env, long_keys)?;
            Ok(longs
                .into_iter()
                .map(|k| ScalarValue::Int32(Some(k as i32)))
                .collect())
        }
        "utf8" => {
            let n = env.get_array_length(string_keys)?;
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let elem: JString = env.get_object_array_element(string_keys, i)?.into();
                let s: String = elem.extract(env)?;
                out.push(ScalarValue::Utf8(Some(s)));
            }
            Ok(out)
        }
        other => Err(Error::input_error(format!(
            "unsupported keyType '{other}'; expected one of int64, int32, utf8"
        ))),
    }
}

fn read_long_array(env: &mut JNIEnv, arr: &JLongArray) -> Result<Vec<i64>> {
    let n = env.get_array_length(arr)?;
    let mut buf = vec![0i64; n as usize];
    if n > 0 {
        env.get_long_array_region(arr, 0, &mut buf)?;
    }
    Ok(buf)
}

// ---- fetch rows ----------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeFetchRows<'local>(
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

    // Decode row_indices (must be parallel to file_paths)
    let row_indices_obj = unsafe { JLongArray::from_raw(row_indices) };
    let n_rows = env.get_array_length(&row_indices_obj)?;
    if n_rows != n_paths {
        return Err(Error::input_error(format!(
            "fetchRows: row_indices length {n_rows} != file_paths length {n_paths}"
        )));
    }
    let mut rids_i64: Vec<jlong> = vec![0; n_rows as usize];
    if n_rows > 0 {
        env.get_long_array_region(&row_indices_obj, 0, &mut rids_i64)?;
    }

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

    let batch = RT.block_on(async { idx.fetch_rows(&row_keys, &proj_refs).await })?;
    record_batch_to_ipc_jbytes(env, &batch)
}

// ---- accessors -----------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeNumFiles(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    match handle_to_idx(handle) {
        Ok(idx) => idx.num_files() as jint,
        Err(_) => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_external_ExternalBtreeIndex_nativeKeyColumn<'local>(
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
    match env.new_string(idx.key_column()) {
        Ok(s) => s.into(),
        Err(e) => {
            Error::runtime_error(format!("new_string: {e}")).throw(&mut env);
            JObject::null()
        }
    }
}
