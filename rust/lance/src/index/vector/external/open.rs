// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `open()` implementation for [`super::ExternalIvfPqIndex`].
//!
//! Reads:
//!
//! - `<index_dir>/<index_uuid>/manifest.json` — parquet file list + build params
//! - `<index_dir>/<index_uuid>/index.idx` — IVF model + PQ codebooks (Lance protobuf format)
//!
//! `index_dir` is the URI passed to `open()`. The single-uuid layout means the
//! caller passes the directory that contains `manifest.json` directly; for now we
//! assume the `<uuid>/` segment is included in the URI. Production callers will
//! get the URI back from `build()` and just round-trip it.

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_index::pb;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Reader;
use lance_io::utils::{read_message, read_metadata_offset};
use lance_linalg::distance::DistanceType;
use object_store::path::Path;

use super::manifest::{ExternalIndexManifest, read_manifest};
use super::parquet_source::ParquetMetaCache;
use super::rerank::{RerankReader, rerank_meta};

/// Constants. `INDEX_FILE_NAME` mirrors what the build path writes.
pub const INDEX_FILE_NAME: &str = "index.idx";

/// All deserialized state of an opened external index, *except* per-partition
/// posting-list data (which the search path lazy-loads). Owns its own
/// [`ObjectStore`] handle plus the index file path so search and fetch_rows can
/// re-issue I/O without re-resolving URIs.
///
/// `object_store` and `index_dir` aren't read by the current search/fetch
/// paths (which open source parquet files directly via `std::fs`); they're
/// kept on the handle for the JNI / remote-store path that lands with #33.
#[allow(dead_code)]
pub struct OpenedExternalIndex {
    pub manifest: ExternalIndexManifest,
    pub ivf: IvfModel,
    pub pq: ProductQuantizer,
    pub metric: DistanceType,
    pub object_store: Arc<ObjectStore>,
    pub index_dir: Path,
    pub index_file_reader: Arc<dyn Reader>,
    /// Per-source-parquet metadata cache shared by the search refinement and
    /// `fetch_rows` paths. Populated lazily on first read of each file; reused
    /// across all subsequent queries on this handle. Skips URI resolution +
    /// `head` + footer/page-index fetch on every cached read — the three
    /// per-call abfss/S3 round trips that dominate per-query latency.
    pub(super) parquet_meta_cache: Arc<ParquetMetaCache>,
    /// Opened SQ8 rerank sidecar, when the index was built with one. Present →
    /// refinement reranks against co-located int8 codes instead of re-reading
    /// the source parquet vector column. `None` → the parquet refine path.
    pub(super) rerank: Option<RerankReader>,
}

/// Resolve `uri` into `(ObjectStore, Path)`, then load both the manifest and the
/// index file's metadata.
pub async fn open_index(uri: &str) -> Result<OpenedExternalIndex> {
    let (object_store, index_dir) = ObjectStore::from_uri(uri).await?;

    let manifest = read_manifest(&object_store, &index_dir).await?;

    let index_file_path = index_dir.clone().join(INDEX_FILE_NAME);
    let reader: Arc<dyn Reader> = Arc::from(object_store.open(&index_file_path).await?);

    // Tail layout (see write_magics): u64 offset, i16 major, i16 minor, 8-byte magic.
    let file_size = reader.size().await?;
    if file_size < 20 {
        return Err(Error::io(format!(
            "external index file at {index_file_path} is too small to contain footer"
        )));
    }
    let block_size = reader.block_size().min(file_size);
    let tail_start = file_size.saturating_sub(block_size.max(20));
    let tail = reader.get_range(tail_start..file_size).await?;
    let metadata_offset = read_metadata_offset(&tail)?;

    let pb_index: pb::Index = read_message(reader.as_ref(), metadata_offset).await?;
    let (ivf, pq, metric) = decode_pb_index(&pb_index)?;

    // Open the rerank sidecar (sq8 or flat) if the manifest records one. Guard that
    // its stored dimension matches the index's — a mismatch means a corrupt or
    // mis-paired sidecar and would silently mis-rank, so fail fast at open.
    let rerank = match rerank_meta(&manifest) {
        Some(meta) => {
            let store = RerankReader::open(&object_store, &index_dir, meta.clone()).await?;
            let index_dim = ivf.dimension();
            if store.dim() != index_dim {
                return Err(Error::index(format!(
                    "rerank store dim {} != index dim {index_dim}",
                    store.dim()
                )));
            }
            Some(store)
        }
        None => None,
    };

    Ok(OpenedExternalIndex {
        manifest,
        ivf,
        pq,
        metric,
        object_store,
        index_dir,
        index_file_reader: reader,
        parquet_meta_cache: Arc::new(ParquetMetaCache::new()),
        rerank,
    })
}

fn decode_pb_index(pb_index: &pb::Index) -> Result<(IvfModel, ProductQuantizer, DistanceType)> {
    use lance_index::pb::vector_index_stage::Stage;
    let vec_idx = match pb_index.implementation.as_ref() {
        Some(lance_index::pb::index::Implementation::VectorIndex(v)) => v,
        _ => {
            return Err(Error::index(
                "external index file is not a VectorIndex".to_string(),
            ));
        }
    };
    let metric: DistanceType =
        lance_index::pb::VectorMetricType::try_from(vec_idx.metric_type)?.into();

    let mut ivf: Option<IvfModel> = None;
    let mut pq: Option<ProductQuantizer> = None;
    for stage in &vec_idx.stages {
        match stage.stage.as_ref() {
            Some(Stage::Ivf(ivf_pb)) => {
                ivf = Some(IvfModel::try_from(ivf_pb.clone())?);
            }
            Some(Stage::Pq(pq_pb)) => {
                pq = Some(ProductQuantizer::from_proto(pq_pb, metric)?);
            }
            Some(Stage::Transform(_)) => {
                // Transform stages are recorded for downstream pipelines but the
                // external builder writes none today; ignore.
            }
            Some(other) => {
                return Err(Error::index(format!(
                    "external index file has unsupported stage: {other:?}"
                )));
            }
            None => {}
        }
    }

    let ivf = ivf.ok_or_else(|| Error::index("external index missing IVF stage"))?;
    let pq = pq.ok_or_else(|| Error::index("external index missing PQ stage"))?;
    Ok((ivf, pq, metric))
}
