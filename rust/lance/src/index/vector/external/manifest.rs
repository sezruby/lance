// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! External index manifest sidecar.
//!
//! `manifest.json` lives next to `index.idx` and records:
//!
//! - The list of parquet files the index covers (encoded `file_id` is implicit
//!   in array position)
//! - Build params (metric, num_partitions, num_sub_vectors, etc.) — informational
//!
//! Sidecar JSON rather than a protobuf field on the index because (a) it lets the
//! file list evolve independently of Lance's format spec, and (b) it's easy to
//! inspect for debugging.

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use super::params::ExternalIvfPqIndexParams;
use super::types::ParquetFileSpec;

pub const MANIFEST_FILE_NAME: &str = "manifest.json";

/// Sidecar file holding the SQ8 (int8 scalar-quantized) rerank store, when present.
pub const RERANK_SQ8_FILE_NAME: &str = "rerank.sq8";

/// Sidecar file holding the full-precision (f32) rerank store, when present.
pub const RERANK_FLAT_FILE_NAME: &str = "rerank.flat";

/// Persistent manifest written alongside `index.idx`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalIndexManifest {
    /// Schema version of this manifest format. Bump on incompatible changes.
    pub manifest_version: u32,
    /// Vector column the index was built over.
    pub vector_column: String,
    /// Parquet files registered with this index. The encoded rid is
    /// `(position_in_this_list_u32 << 32) | row_index_u32`.
    pub files: Vec<ManifestFileEntry>,
    /// Build params, informational. The runtime metric type comes from the index
    /// file's protobuf header; this is a paper trail.
    pub params: ManifestParams,
    /// Metadata for the co-located rerank store, or `None` when the index was
    /// built without one. `#[serde(default)]` keeps pre-rerank manifests readable.
    #[serde(default)]
    pub rerank: Option<RerankStoreMeta>,
}

/// Describes a co-located rerank sidecar.
///
/// Layout is row-major by *global ordinal* — `global_base(file_id) + row_in_file`,
/// where `global_base` is the prefix-sum of `files[..file_id].num_rows`. Each row
/// occupies exactly `bytes_per_row` bytes, so any rid maps to the byte range
/// `[ordinal * bytes_per_row, (ordinal + 1) * bytes_per_row)`. No per-row framing.
///
/// Two kinds share this layout:
/// - `"sq8"`: one `u8` per dimension (`bytes_per_row = dim`), reranked with integer
///   `l2_u8` after quantizing the query with the recorded `bounds`.
/// - `"flat"`: full-precision f32 (`bytes_per_row = dim * 4`), reranked with exact f32
///   L2. `bounds` are unused (`None`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankStoreMeta {
    /// Store kind: `"sq8"` or `"flat"`.
    pub kind: String,
    /// Vector dimension.
    pub dim: usize,
    /// Total rows in the store = sum of `files[*].num_rows`.
    pub total_rows: u64,
    /// Scalar-quantization bounds `[min, max]` used to map f32 → u8. `Some` for the
    /// `"sq8"` kind; `None` for `"flat"` (no quantization). `#[serde(default)]` keeps
    /// older sq8-only manifests readable.
    #[serde(default)]
    pub bounds_min: Option<f64>,
    #[serde(default)]
    pub bounds_max: Option<f64>,
    /// Sidecar layout. Empty (`#[serde(default)]`) → the single-file layout: one
    /// `<index_dir>/<kind.file_name()>` covering all `total_rows` in global-ordinal
    /// order (what the driver-side build writes). Non-empty → a distributed build
    /// wrote one shard file per executor; each entry covers a contiguous global
    /// ordinal range `[ordinal_base, ordinal_base + ordinal_count)`. The reader
    /// routes each candidate ordinal to the shard whose range contains it.
    #[serde(default)]
    pub shards: Vec<RerankShard>,
}

/// One shard file of a distributed rerank sidecar, covering a contiguous global
/// ordinal range. `path` is a full URI (executors write to shared storage).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankShard {
    pub path: String,
    pub ordinal_base: u64,
    pub ordinal_count: u64,
}

impl RerankStoreMeta {
    /// Bytes each stored row occupies: `dim` for sq8, `dim * 4` for flat f32.
    pub fn bytes_per_row(&self) -> usize {
        match self.kind.as_str() {
            "flat" => self.dim * 4,
            _ => self.dim, // sq8: one u8 per dim
        }
    }

    /// True when the sidecar is split into per-executor shard files (distributed
    /// build) rather than a single driver-written file.
    pub fn is_sharded(&self) -> bool {
        !self.shards.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestFileEntry {
    pub file_path: String,
    pub num_rows: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestParams {
    pub num_partitions: usize,
    pub num_sub_vectors: usize,
    pub num_bits_per_sub_vector: usize,
    pub metric: String, // L2 / Cosine / Dot — the textual form from MetricType::to_string
    pub max_iters: usize,
    pub sample_rate: usize,
    pub seed: u64,
}

impl ExternalIndexManifest {
    pub fn from_build(
        vector_column: &str,
        files: &[ParquetFileSpec],
        params: &ExternalIvfPqIndexParams,
        rerank: Option<RerankStoreMeta>,
    ) -> Self {
        Self {
            manifest_version: 1,
            vector_column: vector_column.to_string(),
            files: files
                .iter()
                .map(|s| ManifestFileEntry {
                    file_path: s.file_path.clone(),
                    num_rows: s.num_rows,
                })
                .collect(),
            params: ManifestParams {
                num_partitions: params.num_partitions,
                num_sub_vectors: params.num_sub_vectors,
                num_bits_per_sub_vector: params.num_bits_per_sub_vector,
                metric: format!("{}", params.metric),
                max_iters: params.max_iters,
                sample_rate: params.sample_rate,
                seed: params.seed,
            },
            rerank,
        }
    }

    /// Resolve a manifest entry's `file_id`. Currently O(n) which is fine for
    /// typical file counts (< 10k).
    pub fn file_id(&self, file_path: &str) -> Option<u32> {
        self.files
            .iter()
            .position(|e| e.file_path == file_path)
            .map(|i| i as u32)
    }

    pub fn file_path(&self, file_id: u32) -> Option<&str> {
        self.files
            .get(file_id as usize)
            .map(|e| e.file_path.as_str())
    }

    /// Global ordinal of `file_id`'s first row in the row-major rerank store:
    /// the prefix-sum of preceding files' `num_rows`. `None` if `file_id` is out
    /// of range. A rid `(file_id, row_in_file)` maps to global ordinal
    /// `global_base(file_id)? + row_in_file`.
    pub fn global_base(&self, file_id: u32) -> Option<u64> {
        if file_id as usize > self.files.len() {
            return None;
        }
        Some(
            self.files[..file_id as usize]
                .iter()
                .map(|e| e.num_rows)
                .sum(),
        )
    }
}

/// Write the manifest to `dir/manifest.json`.
pub async fn write_manifest(
    object_store: &ObjectStore,
    dir: &Path,
    manifest: &ExternalIndexManifest,
) -> Result<()> {
    let path = dir.clone().join(MANIFEST_FILE_NAME);
    let json = serde_json::to_vec_pretty(manifest)
        .map_err(|e| Error::io(format!("manifest serialize failed: {e}")))?;
    object_store
        .put(&path, &json)
        .await
        .map_err(|e| Error::io(format!("manifest write failed at {path}: {e}")))?;
    Ok(())
}

/// Read `dir/manifest.json`.
pub async fn read_manifest(
    object_store: &ObjectStore,
    dir: &Path,
) -> Result<ExternalIndexManifest> {
    let path = dir.clone().join(MANIFEST_FILE_NAME);
    let bytes = object_store
        .read_one_all(&path)
        .await
        .map_err(|e| Error::io(format!("manifest read failed at {path}: {e}")))?;
    let manifest: ExternalIndexManifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::io(format!("manifest parse failed at {path}: {e}")))?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_linalg::distance::MetricType;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn from_build_round_trips_via_json() {
        let files = vec![
            ParquetFileSpec::with_metadata(
                "/tmp/a.parquet",
                100,
                Arc::new(arrow_schema::Schema::empty()),
            ),
            ParquetFileSpec::with_metadata(
                "/tmp/b.parquet",
                200,
                Arc::new(arrow_schema::Schema::empty()),
            ),
        ];
        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(64)
            .num_sub_vectors(8)
            .metric(MetricType::Cosine)
            .build();

        let manifest = ExternalIndexManifest::from_build("emb", &files, &params, None);
        let json = serde_json::to_vec_pretty(&manifest).unwrap();
        let parsed: ExternalIndexManifest = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed.manifest_version, 1);
        assert_eq!(parsed.vector_column, "emb");
        assert_eq!(parsed.files.len(), 2);
        assert_eq!(parsed.file_id("/tmp/b.parquet"), Some(1));
        assert_eq!(parsed.file_path(0), Some("/tmp/a.parquet"));
        assert_eq!(parsed.file_path(99), None);
        assert_eq!(parsed.params.num_partitions, 64);
        assert_eq!(parsed.params.metric, "cosine");
        // Default build carries no rerank store.
        assert!(parsed.rerank.is_none());
        // global_base is the prefix-sum of preceding files' num_rows.
        assert_eq!(parsed.global_base(0), Some(0));
        assert_eq!(parsed.global_base(1), Some(100));
        assert_eq!(parsed.global_base(2), Some(300));
        assert_eq!(parsed.global_base(3), None);
    }

    #[test]
    fn rerank_meta_round_trips_and_is_back_compat() {
        let files = vec![ParquetFileSpec::with_metadata(
            "/tmp/a.parquet",
            10,
            Arc::new(arrow_schema::Schema::empty()),
        )];
        let params = ExternalIvfPqIndexParams::builder().build();
        let rerank = Some(RerankStoreMeta {
            kind: "sq8".to_string(),
            dim: 8,
            total_rows: 10,
            bounds_min: Some(-1.5),
            bounds_max: Some(2.5),
            shards: Vec::new(),
        });
        let manifest = ExternalIndexManifest::from_build("emb", &files, &params, rerank);
        let json = serde_json::to_vec_pretty(&manifest).unwrap();
        let parsed: ExternalIndexManifest = serde_json::from_slice(&json).unwrap();
        let r = parsed.rerank.expect("rerank meta round-trips");
        assert_eq!(r.kind, "sq8");
        assert_eq!(r.dim, 8);
        assert_eq!(r.total_rows, 10);
        assert_eq!(r.bounds_min, Some(-1.5));
        assert_eq!(r.bounds_max, Some(2.5));
        assert_eq!(r.bytes_per_row(), 8); // sq8: one u8 per dim
        assert!(!r.is_sharded());

        // A "flat" store: full-precision, no bounds, dim*4 bytes/row.
        let flat = RerankStoreMeta {
            kind: "flat".to_string(),
            dim: 8,
            total_rows: 10,
            bounds_min: None,
            bounds_max: None,
            shards: Vec::new(),
        };
        assert_eq!(flat.bytes_per_row(), 32);

        // A manifest JSON produced before the rerank field existed must still
        // parse (serde default → None), so existing indexes keep opening.
        let legacy = r#"{
            "manifest_version": 1,
            "vector_column": "emb",
            "files": [{"file_path": "/tmp/a.parquet", "num_rows": 10}],
            "params": {
                "num_partitions": 256, "num_sub_vectors": 16,
                "num_bits_per_sub_vector": 8, "metric": "l2",
                "max_iters": 50, "sample_rate": 256, "seed": 0
            }
        }"#;
        let parsed_legacy: ExternalIndexManifest = serde_json::from_str(legacy).unwrap();
        assert!(parsed_legacy.rerank.is_none());
    }

    #[tokio::test]
    async fn write_then_read() {
        let tmp = TempDir::new().unwrap();
        let uri = tmp.path().to_str().unwrap();
        let (object_store, root) = ObjectStore::from_uri(uri).await.unwrap();

        let files = vec![ParquetFileSpec::with_metadata(
            "/tmp/x.parquet",
            42,
            Arc::new(arrow_schema::Schema::empty()),
        )];
        let params = ExternalIvfPqIndexParams::builder().build();
        let manifest = ExternalIndexManifest::from_build("vec", &files, &params, None);
        write_manifest(&object_store, &root, &manifest)
            .await
            .unwrap();

        let parsed = read_manifest(&object_store, &root).await.unwrap();
        assert_eq!(parsed.vector_column, "vec");
        assert_eq!(parsed.files.len(), 1);
        assert_eq!(parsed.files[0].num_rows, 42);
    }
}
