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

        let manifest = ExternalIndexManifest::from_build("emb", &files, &params);
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
        let manifest = ExternalIndexManifest::from_build("vec", &files, &params);
        write_manifest(&object_store, &root, &manifest)
            .await
            .unwrap();

        let parsed = read_manifest(&object_store, &root).await.unwrap();
        assert_eq!(parsed.vector_column, "vec");
        assert_eq!(parsed.files.len(), 1);
        assert_eq!(parsed.files[0].num_rows, 42);
    }
}
