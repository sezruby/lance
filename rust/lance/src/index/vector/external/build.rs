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
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_index::vector::ivf::IvfTransformer;
use lance_index::vector::ivf::shuffler::shuffle_dataset;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::kmeans::KMeans;
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

use super::manifest::{ExternalIndexManifest, RerankStoreMeta, write_manifest};
use super::params::{ExternalIvfPqIndexParams, RerankStore};
use super::parquet_source::ParquetVectorSource;
use super::rerank;
use super::rerank::build_store;
use super::types::ParquetFileSpec;
use crate::index::vector::ivf::write_ivf_pq_file_external;
use lance_index::vector::sq::ScalarQuantizer;
use lance_linalg::distance::MetricType;

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

    let source = ParquetVectorSource::try_new(files.clone(), vector_column).await?;

    // 1+2. Train the IVF centroids + PQ codebook on a sample. These are the only
    // steps that need a corpus-wide view; both are cheap (sample-sized).
    let (ivf, pq, training) = train_quantizers(&source, &params).await?;

    // 3. Assign + PQ-encode every row over the whole corpus into partition-binned
    // batches (offset 0 = whole corpus), then wrap as streams for the writer.
    let collected = shard_partition_streams(&source, vector_column, &ivf, &pq, &params).await?;
    let partitioned_streams = batches_to_streams(collected);

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
    // manifest records authoritative values. The rerank store's global-ordinal
    // layout depends on these counts, so they must be authoritative before it
    // (or the manifest) is written.
    let mut resolved_files = files;
    for spec in resolved_files.iter_mut() {
        if spec.num_rows == 0 {
            spec.num_rows = read_parquet_num_rows(&spec.file_path).await?;
        }
    }

    // Optionally build the co-located rerank store. For SQ8, bounds are learned
    // from the same training sample already used for kmeans/PQ (no extra sampling
    // pass); Flat needs no bounds. Either way the store is written from a full scan
    // in manifest order, giving the row-major-by-global-ordinal layout the search
    // path maps rids onto.
    let rerank_kind = match params.rerank_store {
        RerankStore::None => None,
        RerankStore::Sq8 => Some(rerank::RerankKind::Sq8),
        RerankStore::Flat => Some(rerank::RerankKind::Flat),
    };
    let rerank_meta: Option<RerankStoreMeta> = match rerank_kind {
        None => None,
        Some(kind) => {
            if matches!(params.metric, MetricType::Dot) {
                return Err(Error::invalid_input(
                    "rerank store supports L2 / Cosine metrics only, not Dot",
                ));
            }
            let dim = training.value_length() as usize;
            // Bounds are only meaningful for SQ8; learn them from the training
            // sample. Flat ignores them.
            let bounds = match kind {
                rerank::RerankKind::Sq8 => {
                    let mut sq = ScalarQuantizer::new(8, dim);
                    sq.update_bounds::<arrow::datatypes::Float32Type>(&training)?;
                    sq.bounds()
                }
                rerank::RerankKind::Flat => 0.0..0.0,
            };
            let meta = build_store(
                &object_store,
                &index_dir,
                &resolved_files,
                vector_column,
                dim,
                kind,
                bounds,
            )
            .await?;
            Some(meta)
        }
    };

    let manifest =
        ExternalIndexManifest::from_build(vector_column, &resolved_files, &params, rerank_meta);
    write_manifest(&object_store, &index_dir, &manifest).await?;

    Ok(index_uuid)
}

async fn read_parquet_num_rows(path: &str) -> Result<u64> {
    let builder = super::parquet_source::open_parquet_async(path).await?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}

/// Test-only: construct a whole-corpus source so a test can `train_quantizers`
/// once and share the result across shard-count variations.
#[cfg(test)]
pub(super) async fn _test_source(
    files: Vec<ParquetFileSpec>,
    vector_column: &str,
) -> ParquetVectorSource {
    ParquetVectorSource::try_new(files, vector_column)
        .await
        .expect("test source")
}

/// Train the IVF centroids + PQ codebook from a sample of `source`. These are the
/// only corpus-wide steps in the build; they run on a sample (`num_partitions *
/// sample_rate` rows) so they are cheap even at large |R|. Returns the trained
/// models plus the training sample (reused for SQ8 bounds).
///
/// In a distributed build this runs once on the driver; the returned `(ivf, pq)`
/// are serialized to protobuf and broadcast to the executors, each of which feeds
/// them to [`shard_partition_streams`] over its file shard.
pub(super) async fn train_quantizers(
    source: &ParquetVectorSource,
    params: &ExternalIvfPqIndexParams,
) -> Result<(IvfModel, ProductQuantizer, FixedSizeListArray)> {
    let sample_size = (params.num_partitions * params.sample_rate)
        .min(source.num_rows().await? as usize)
        .max(params.num_partitions);
    let training = source.sample(sample_size).await?;
    let kmeans = KMeans::new(&training, params.num_partitions, params.max_iters as u32)
        .map_err(|e| Error::index(format!("kmeans training failed: {e}")))?;
    let centroids =
        FixedSizeListArray::try_new_from_values(kmeans.centroids.clone(), kmeans.dimension as i32)
            .map_err(|e| Error::index(format!("kmeans centroids → FixedSizeListArray: {e}")))?;
    let ivf = IvfModel::new(centroids.clone(), None);

    // Train the PQ codebook on RESIDUALS (vector − assigned centroid), not raw vectors,
    // for L2/Cosine. This mirrors Lance-core's IVF-PQ build (`build_pq_model` computes the
    // residual of the training sample before `PQBuildParams::build`). It is required for
    // correctness, not just quality: the encode path (`IvfTransformer::with_pq`) auto-inserts
    // a ResidualTransform for these metrics, so PQ codes are applied in residual space —
    // training the codebook on raw vectors would mismatch train vs encode distributions and
    // materially drops recall. Dot product does not use residuals (matches `use_residual`).
    let use_residual = matches!(params.metric, MetricType::L2 | MetricType::Cosine);
    let pq_training = if use_residual {
        let ivf_transformer =
            lance_index::vector::ivf::new_ivf_transformer(centroids, MetricType::L2, vec![]);
        ivf_transformer.compute_residual(&training)?
    } else {
        training.clone()
    };
    let pq = PQBuildParams::new(params.num_sub_vectors, params.num_bits_per_sub_vector)
        .build(&pq_training, params.metric)
        .map_err(|e| Error::index(format!("PQ training failed: {e}")))?;
    Ok((ivf, pq, training))
}

/// Assign + PQ-encode every row of `source` into partition-binned streams, using
/// the pre-trained `ivf` + `pq`. This is the per-row, embarrassingly-parallel work
/// (the 43 GB scan): given the shared quantizers it needs no corpus-wide state, so
/// each executor can run it over its file shard and the driver merges the results
/// by concatenating the returned stream vecs (each stream is sorted by part_id;
/// [`write_ivf_pq_file_external`]'s heap-merge unions same-part_id streams across
/// shards).
///
/// `source` carries the shard's `file_id_offset`, so the emitted `_rowid`s are
/// globally consistent across shards.
pub(super) async fn shard_partition_streams(
    source: &ParquetVectorSource,
    vector_column: &str,
    ivf: &IvfModel,
    pq: &ProductQuantizer,
    params: &ExternalIvfPqIndexParams,
) -> Result<Vec<Vec<arrow_array::RecordBatch>>> {
    let centroids = ivf
        .centroids_array()
        .ok_or_else(|| Error::index("ivf model has no centroids"))?
        .clone();
    let transformer = Arc::new(IvfTransformer::with_pq(
        centroids,
        params.metric.into(),
        vector_column,
        pq.clone(),
        None,
    ));
    let raw_stream = source.iter_batches().await?;
    let streams = shuffle_dataset(
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

    // Drain each partition stream to owned batches. The shuffle already spilled to
    // files, so this holds one partition's rows at a time, not the whole shard.
    // Returning owned batches frees the borrow on `ivf`/`pq` so the caller can move
    // them into the writer, and lets the distributed path ship these per-shard.
    // Empty partitions (no rows in this shard) are dropped — the writer's heap-seed
    // reads batch[0], so empty streams must not reach it.
    let mut out: Vec<Vec<arrow_array::RecordBatch>> = Vec::new();
    for s in streams {
        let batches: Vec<arrow_array::RecordBatch> = s.try_collect().await?;
        if !batches.is_empty() {
            out.push(batches);
        }
    }
    Ok(out)
}

/// Wrap collected partition batch-vecs into owned streams for
/// [`write_ivf_pq_file_external`].
fn batches_to_streams(
    collected: Vec<Vec<arrow_array::RecordBatch>>,
) -> Vec<impl futures::Stream<Item = Result<arrow_array::RecordBatch>>> {
    collected
        .into_iter()
        .map(|batches| futures::stream::iter(batches.into_iter().map(Ok::<_, Error>)))
        .collect()
}

/// Assemble a final index from independently-built shards, in-process. This mirrors
/// the distributed build's driver-side merge WITHOUT Spark: train once, split the
/// file list into `num_shards` contiguous groups, run [`shard_partition_streams`]
/// per shard with the correct `file_id_offset`, concatenate every shard's
/// per-partition streams into one Vec, and hand them to
/// [`write_ivf_pq_file_external`] (whose heap-merge unions same-part_id streams
/// across shards). The rerank sidecar is built over the full ordered file list.
///
/// Its only purpose is to PROVE the sharded path produces an index identical to the
/// single-box build (see the `distributed_matches_single_box` test). The real
/// distributed build runs the same three phases across executors via JNI; this
/// function is the local correctness oracle for that design.
pub(super) async fn build_index_from_shards_local(
    files: Vec<ParquetFileSpec>,
    vector_column: &str,
    output_uri: &str,
    params: ExternalIvfPqIndexParams,
    num_shards: usize,
    // Pre-trained quantizers to reuse. `None` → train fresh over the corpus.
    // Tests pass `Some(..)` so a 1-shard and an N-shard build share IDENTICAL
    // centroids/codebook, isolating the sharding+merge from kmeans nondeterminism.
    pretrained: Option<(IvfModel, ProductQuantizer, FixedSizeListArray)>,
) -> Result<Uuid> {
    if files.is_empty() {
        return Err(Error::invalid_input(
            "build requires at least one parquet file",
        ));
    }
    let num_shards = num_shards.clamp(1, files.len());

    // Phase 1 (driver): train once over the whole corpus (or reuse pre-trained).
    let (ivf, pq, training) = match pretrained {
        Some(t) => t,
        None => {
            let full_source = ParquetVectorSource::try_new(files.clone(), vector_column).await?;
            train_quantizers(&full_source, &params).await?
        }
    };

    // Phase 2 (executors, simulated): each shard assigns+encodes its files using the
    // shared quantizers, tagging rids with its global file_id_offset. Each shard's
    // partition streams are drained to in-memory batch vecs before the shard source
    // drops — this also mirrors the real distributed path, where an executor
    // materializes its shard's partition-binned output and ships it to the driver.
    // Each collected vec is one partition-sorted stream; concatenating across shards
    // gives the writer's heap-merge every same-part_id stream to union.
    let mut collected: Vec<Vec<arrow_array::RecordBatch>> = Vec::new();
    let shard_size = files.len().div_ceil(num_shards);
    let mut offset = 0usize;
    while offset < files.len() {
        let end = (offset + shard_size).min(files.len());
        let shard_files = files[offset..end].to_vec();
        let shard_source =
            ParquetVectorSource::try_new_with_offset(shard_files, vector_column, offset as u32)
                .await?;
        collected.extend(
            shard_partition_streams(&shard_source, vector_column, &ivf, &pq, &params).await?,
        );
        offset = end;
    }
    let all_streams = batches_to_streams(collected);

    // Phase 3 (driver): merge every shard's streams into one index file. The
    // heap-merge in write_pq_partitions unions all streams sharing a part_id, so a
    // flat concatenation across shards is exactly the merge.
    let (object_store, root_path) = ObjectStore::from_uri(output_uri).await?;
    let index_uuid = Uuid::new_v4();
    let index_dir = root_path.clone().join(index_uuid.to_string());
    let index_path = index_dir.clone().join(super::open::INDEX_FILE_NAME);
    write_ivf_pq_file_external(
        &object_store,
        &index_path,
        vector_column,
        "external_ivf_pq",
        0,
        ivf,
        pq,
        all_streams,
    )
    .await?;

    // Resolve num_rows + build sidecar + manifest, same as the whole-corpus path.
    let mut resolved_files = files;
    for spec in resolved_files.iter_mut() {
        if spec.num_rows == 0 {
            spec.num_rows = read_parquet_num_rows(&spec.file_path).await?;
        }
    }
    let rerank_kind = match params.rerank_store {
        RerankStore::None => None,
        RerankStore::Sq8 => Some(rerank::RerankKind::Sq8),
        RerankStore::Flat => Some(rerank::RerankKind::Flat),
    };
    let rerank_meta: Option<RerankStoreMeta> = match rerank_kind {
        None => None,
        Some(kind) => {
            let dim = training.value_length() as usize;
            let bounds = match kind {
                rerank::RerankKind::Sq8 => {
                    let mut sq = ScalarQuantizer::new(8, dim);
                    sq.update_bounds::<arrow::datatypes::Float32Type>(&training)?;
                    sq.bounds()
                }
                rerank::RerankKind::Flat => 0.0..0.0,
            };
            Some(
                build_store(
                    &object_store,
                    &index_dir,
                    &resolved_files,
                    vector_column,
                    dim,
                    kind,
                    bounds,
                )
                .await?,
            )
        }
    };
    let manifest =
        ExternalIndexManifest::from_build(vector_column, &resolved_files, &params, rerank_meta);
    write_manifest(&object_store, &index_dir, &manifest).await?;
    Ok(index_uuid)
}
