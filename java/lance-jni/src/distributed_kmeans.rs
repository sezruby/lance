// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JNI shim for the scheduler-neutral distributed IVF centroid-training primitives.
//!
//! Lifted from the `lance_index::vector::kmeans::distributed` layer. Callers (Spark,
//! custom RPC) own broadcast, tree-reduce, and convergence; this module exposes only
//! the math: `mergePartialStats`, `reducePartialStats`, `finalizeCentroids`,
//! `selectInitialCentroids`, `bootstrapCentroids`.
//!
//! Every native moves Arrow data across the boundary as IPC `byte[]`. Centroid FSLs
//! round-trip through a single-column ("vec") IPC batch that preserves the inner
//! Float16/Float32/Float64 dtype; `PartialStats` round-trips as its own record batch.
//! The external-parquet training seams (sampling, in-memory E-step, payload assembly)
//! that build on these primitives live on `ExternalIvfPqIndex` (see `external_index`),
//! which reuses the marshalling helpers below.

use crate::error::{Error, Result};
use crate::external_index::parse_metric;

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use arrow_schema::{Field, Schema};
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObjectArray, JString};
use jni::sys::{jbyteArray, jint, jlong};
use std::sync::Arc;

use lance_index::vector::kmeans::distributed as l1;

pub(crate) fn arrow_err(e: arrow::error::ArrowError) -> Error {
    Error::input_error(e.to_string())
}

pub(crate) fn record_batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).map_err(arrow_err)?;
        writer.write(batch).map_err(arrow_err)?;
        writer.finish().map_err(arrow_err)?;
    }
    Ok(buf)
}

pub(crate) fn ipc_to_record_batch(env: &mut JNIEnv, jba: &JByteArray) -> Result<RecordBatch> {
    let bytes = env.convert_byte_array(jba)?;
    let mut reader = StreamReader::try_new(std::io::Cursor::new(bytes), None).map_err(arrow_err)?;
    reader
        .next()
        .ok_or_else(|| Error::input_error("empty IPC stream".to_string()))?
        .map_err(arrow_err)
}

/// Read a single-column IPC batch back into the centroid/sample FSL, validating the
/// inner dtype is a float kind (the kmeans primitives only accept Float16/32/64).
pub(crate) fn ipc_to_centroids_fsl(
    env: &mut JNIEnv,
    jba: &JByteArray,
) -> Result<FixedSizeListArray> {
    let batch = ipc_to_record_batch(env, jba)?;
    if batch.num_columns() != 1 {
        return Err(Error::input_error(format!(
            "FSL IPC must have a single column, got {}",
            batch.num_columns()
        )));
    }
    let fsl = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::input_error("column must be FixedSizeList".to_string()))?
        .clone();
    if !matches!(
        fsl.value_type(),
        arrow_schema::DataType::Float16
            | arrow_schema::DataType::Float32
            | arrow_schema::DataType::Float64
    ) {
        return Err(Error::input_error(format!(
            "FSL inner dtype must be Float16/Float32/Float64, got {}",
            fsl.value_type()
        )));
    }
    Ok(fsl)
}

/// Wrap a centroid/sample FSL into a single-column ("vec") Arrow-IPC payload; the
/// schema preserves the original inner dtype so Float16/Float32/Float64 round-trip
/// unchanged.
pub(crate) fn fsl_to_centroids_ipc(fsl: &FixedSizeListArray) -> Result<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vec",
        fsl.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(fsl.clone())]).map_err(arrow_err)?;
    record_batch_to_ipc(&batch)
}

pub(crate) fn read_byte_array_2d(env: &mut JNIEnv, arr: &JObjectArray) -> Result<Vec<RecordBatch>> {
    let len = env.get_array_length(arr)? as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let element = env.get_object_array_element(arr, i as i32)?;
        let jba: JByteArray = JByteArray::from(element);
        out.push(ipc_to_record_batch(env, &jba)?);
    }
    Ok(out)
}

trait Pipe: Sized {
    fn pipe<R, F: FnOnce(Self) -> R>(self, f: F) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

/// Convert the `Ok(Vec<u8>)` of a native's inner closure into a returned `jbyteArray`,
/// throwing a Java exception on the `Err` path. Shared tail of every `byte[]`-returning
/// native below.
macro_rules! return_bytes {
    ($env:expr, $inner:expr) => {
        crate::ok_or_throw_with_return!($env, $inner, JByteArray::default().into_raw()).pipe(
            |bytes| match $env.byte_array_from_slice(&bytes) {
                Ok(arr) => arr.into_raw(),
                Err(e) => {
                    let _ = $env.throw_new("java/lang/RuntimeException", e.to_string());
                    JByteArray::default().into_raw()
                }
            },
        )
    };
}

// ---- scheduler-neutral kmeans primitives (l1) --------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeMergePartialStats<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    a_ipc: JByteArray<'local>,
    b_ipc: JByteArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let a = l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &a_ipc)?)?;
        let b = l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &b_ipc)?)?;
        let merged = l1::merge_partial_stats(a, b)?;
        record_batch_to_ipc(merged.record_batch())
    };
    return_bytes!(env, inner())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeReducePartialStats<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    stats_arr: JObjectArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let batches = read_byte_array_2d(&mut env, &stats_arr)?;
        let mut parsed = Vec::with_capacity(batches.len());
        for b in batches {
            parsed.push(l1::PartialStats::from_record_batch(b)?);
        }
        let merged = l1::reduce_partial_stats(parsed)?;
        record_batch_to_ipc(merged.record_batch())
    };
    return_bytes!(env, inner())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeFinalizeCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    stats_ipc: JByteArray<'local>,
    prev_ipc: JByteArray<'local>,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        let stats =
            l1::PartialStats::from_record_batch(ipc_to_record_batch(&mut env, &stats_ipc)?)?;
        let prev = ipc_to_centroids_fsl(&mut env, &prev_ipc)?;
        let fsl = l1::finalize_centroids(&stats, &prev)?;
        fsl_to_centroids_ipc(&fsl)
    };
    return_bytes!(env, inner())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeSelectInitialCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    samples_arr: JObjectArray<'local>,
    k: jint,
    rng_seed: jlong,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        if k < 0 {
            return Err(Error::input_error(format!("k must be >= 0, got {}", k)));
        }
        let batches = read_byte_array_2d(&mut env, &samples_arr)?;
        let fsl = l1::select_initial_centroids(batches, k as usize, rng_seed as u64)?;
        fsl_to_centroids_ipc(&fsl)
    };
    return_bytes!(env, inner())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_DistributedKMeans_nativeBootstrapCentroids<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    samples_arr: JObjectArray<'local>,
    k: jint,
    distance_type_jstr: JString<'local>,
    rng_seed: jlong,
) -> jbyteArray {
    let mut inner = || -> Result<Vec<u8>> {
        if k < 0 {
            return Err(Error::input_error(format!("k must be >= 0, got {}", k)));
        }
        let dt_str: String = env.get_string(&distance_type_jstr)?.into();
        let dt = parse_metric(&dt_str)?;
        let batches = read_byte_array_2d(&mut env, &samples_arr)?;
        let fsl = l1::bootstrap_centroids(batches, k as usize, dt, rng_seed as u64)?;
        fsl_to_centroids_ipc(&fsl)
    };
    return_bytes!(env, inner())
}
