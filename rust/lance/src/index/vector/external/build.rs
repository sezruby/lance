// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `build()` implementation for [`super::ExternalIvfPqIndex`].
//!
//! Composes the existing public Lance APIs:
//!
//! 1. [`ParquetVectorSource::sample`] → kmeans (`KMeans::new`) → centroids → `IvfModel::new`
//! 2. [`ParquetVectorSource::sample`] → `PQBuildParams::build` → `ProductQuantizer`
//! 3. [`ParquetVectorSource::iter_batches`] → `IvfTransformer::with_pq` → `shuffle_dataset`
//!    → partition-binned streams
//! 4. `write_ivf_pq_file_external(object_store, path, ..., streams)` writes the index
//!
//! The Dataset dependency is gone end-to-end on this path.

use std::sync::Arc;

use arrow_array::FixedSizeListArray;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_index::vector::ivf::IvfTransformer;
use lance_index::vector::ivf::shuffler::shuffle_dataset;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::kmeans::KMeans;
use lance_index::vector::pq::PQBuildParams;
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

use super::manifest::{ExternalIndexManifest, write_manifest};
use super::params::ExternalIvfPqIndexParams;
use super::parquet_source::ParquetVectorSource;
use super::types::ParquetFileSpec;
use crate::index::vector::ivf::write_ivf_pq_file_external;

/// Internal entry point — drives the full build pipeline.
///
/// Layout written under `output_uri`:
///
/// ```text
/// <output_uri>/
///   <index_uuid>/
///     index.idx     ← IVF-PQ partitions + protobuf metadata
///     manifest.json ← ParquetFileSpec list + build params (Phase 1.4 lands this)
/// ```
///
/// The manifest sidecar lands in Phase 1.4 along with `open()`. For now `build()`
/// writes only `index.idx` so the search-side primitives can be validated.
pub(super) async fn build_index(
    files: Vec<ParquetFileSpec>,
    vector_column: &str,
    output_uri: &str,
    params: ExternalIvfPqIndexParams,
) -> Result<Uuid> {
    if files.is_empty() {
        return Err(Error::invalid_input(
            "build_index requires at least one parquet file",
        ));
    }

    let source = ParquetVectorSource::try_new(files.clone(), vector_column)?;

    // 1. Train kmeans for IVF centroids
    let sample_size = (params.num_partitions * params.sample_rate)
        .min(source.num_rows()? as usize)
        .max(params.num_partitions);
    let training = source.sample(sample_size).await?;
    let kmeans = KMeans::new(&training, params.num_partitions, params.max_iters as u32)
        .map_err(|e| Error::index(format!("kmeans training failed: {e}")))?;

    let centroids =
        FixedSizeListArray::try_new_from_values(kmeans.centroids.clone(), kmeans.dimension as i32)
            .map_err(|e| Error::index(format!("kmeans centroids → FixedSizeListArray: {e}")))?;
    let ivf = IvfModel::new(centroids.clone(), None);

    // 2. Train PQ codebooks over the same training data
    let pq = PQBuildParams::new(params.num_sub_vectors, params.num_bits_per_sub_vector)
        .build(&training, params.metric.into())
        .map_err(|e| Error::index(format!("PQ training failed: {e}")))?;

    // 3. Build IVF transformer; shuffle source batches into partition-binned streams
    let transformer = Arc::new(IvfTransformer::with_pq(
        centroids,
        params.metric.into(),
        vector_column,
        pq.clone(),
        None,
    ));
    let raw_stream = source.iter_batches()?;
    let partitioned_streams = shuffle_dataset(
        raw_stream,
        transformer,
        /* precomputed_partitions = */ None,
        params.num_partitions as u32,
        /* shuffle_partition_batches = */ 1024,
        /* shuffle_partition_concurrency = */ 2,
        /* precomputed_shuffle_buffers = */ None,
    )
    .await
    .map_err(|e| Error::index(format!("shuffle_dataset failed: {e}")))?;

    // 4. Write the index file. Resolve `output_uri` into (ObjectStore, Path) and
    // write under <output_uri>/<uuid>/index.idx.
    let (object_store, root_path) = ObjectStore::from_uri(output_uri).await?;
    let index_uuid = Uuid::new_v4();
    let index_dir = root_path.clone().join(index_uuid.to_string());
    let index_path = index_dir.clone().join(super::open::INDEX_FILE_NAME);

    write_ivf_pq_file_external(
        &object_store,
        &index_path,
        vector_column,
        /* index_name = */ "external_ivf_pq",
        /* dataset_version = */ 0,
        ivf,
        pq,
        partitioned_streams,
    )
    .await?;

    // Resolve any unfilled num_rows on the file specs from their footers so the
    // manifest records authoritative values.
    let mut resolved_files = files;
    for spec in resolved_files.iter_mut() {
        if spec.num_rows == 0 {
            spec.num_rows = read_parquet_num_rows(&spec.file_path)?;
        }
    }
    let manifest = ExternalIndexManifest::from_build(vector_column, &resolved_files, &params);
    write_manifest(&object_store, &index_dir, &manifest).await?;

    Ok(index_uuid)
}

fn read_parquet_num_rows(path: &str) -> Result<u64> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(path)
        .map_err(|e| Error::invalid_input(format!("failed to open {path}: {e}")))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::invalid_input(format!("failed to read footer for {path}: {e}")))?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}
