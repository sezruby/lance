// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! External scalar-index manifest sidecar.
//!
//! `manifest.json` lives next to the BTree index files (`btree_lookup.lance`,
//! `btree_pages.lance`) and records:
//!
//! - The list of parquet files the index covers. The encoded rid is
//!   `(position_in_this_list_u32 << 32) | row_index_u32`, exactly as in the
//!   external vector index — so the two share the row-identity model and the
//!   [`ManifestFileEntry`] type / `file_id` ↔ `file_path` helpers.
//! - The scalar key column the index was built over.
//! - An index-type tag (`"btree"`) so a reader can reject a manifest written for
//!   a different scalar index kind before touching the index files.

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};

// Reuse the vector external index's file-entry record and manifest file name —
// the on-disk file-list shape and the sidecar name are identical across the two.
pub use crate::index::vector::external::manifest::{MANIFEST_FILE_NAME, ManifestFileEntry};

/// Index-type tag stored in the manifest. Guards against opening a BTree manifest
/// as some other scalar index kind (or vice versa) once more land.
pub const INDEX_TYPE_BTREE: &str = "btree";

/// Persistent manifest written alongside the BTree index files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarExternalManifest {
    /// Schema version of this manifest format. Bump on incompatible changes.
    pub manifest_version: u32,
    /// Scalar index kind, e.g. [`INDEX_TYPE_BTREE`].
    pub index_type: String,
    /// Scalar key column the index was built over.
    pub key_column: String,
    /// Parquet files registered with this index. The encoded rid is
    /// `(position_in_this_list_u32 << 32) | row_index_u32`.
    pub files: Vec<ManifestFileEntry>,
}

impl ScalarExternalManifest {
    pub fn from_build(key_column: &str, files: &[crate::index::scalar_external::ParquetFileSpec]) -> Self {
        Self {
            manifest_version: 1,
            index_type: INDEX_TYPE_BTREE.to_string(),
            key_column: key_column.to_string(),
            files: files
                .iter()
                .map(|s| ManifestFileEntry {
                    file_path: s.file_path.clone(),
                    num_rows: s.num_rows,
                })
                .collect(),
        }
    }

    /// Resolve a manifest entry's `file_id` (its position in the file list).
    pub fn file_id(&self, file_path: &str) -> Option<u32> {
        self.files
            .iter()
            .position(|e| e.file_path == file_path)
            .map(|i| i as u32)
    }

    /// File path for a given `file_id` (the high 32 bits of the rid), or `None`
    /// if the id is out of range.
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
    manifest: &ScalarExternalManifest,
) -> Result<()> {
    let path = dir.clone().join(MANIFEST_FILE_NAME);
    let json = serde_json::to_vec_pretty(manifest)
        .map_err(|e| Error::io(format!("scalar manifest serialize failed: {e}")))?;
    object_store
        .put(&path, &json)
        .await
        .map_err(|e| Error::io(format!("scalar manifest write failed at {path}: {e}")))?;
    Ok(())
}

/// Read `dir/manifest.json`.
pub async fn read_manifest(object_store: &ObjectStore, dir: &Path) -> Result<ScalarExternalManifest> {
    let path = dir.clone().join(MANIFEST_FILE_NAME);
    let bytes = object_store
        .read_one_all(&path)
        .await
        .map_err(|e| Error::io(format!("scalar manifest read failed at {path}: {e}")))?;
    let manifest: ScalarExternalManifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::io(format!("scalar manifest parse failed at {path}: {e}")))?;
    if manifest.index_type != INDEX_TYPE_BTREE {
        return Err(Error::invalid_input(format!(
            "scalar manifest at {path} has index_type '{}', expected '{INDEX_TYPE_BTREE}'",
            manifest.index_type
        )));
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::scalar_external::ParquetFileSpec;

    #[test]
    fn from_build_round_trips_via_json() {
        let files = vec![
            ParquetFileSpec::with_metadata(
                "/tmp/a.parquet",
                100,
                std::sync::Arc::new(arrow_schema::Schema::empty()),
            ),
            ParquetFileSpec::with_metadata(
                "/tmp/b.parquet",
                200,
                std::sync::Arc::new(arrow_schema::Schema::empty()),
            ),
        ];
        let manifest = ScalarExternalManifest::from_build("id", &files);
        let json = serde_json::to_vec_pretty(&manifest).unwrap();
        let parsed: ScalarExternalManifest = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed.manifest_version, 1);
        assert_eq!(parsed.index_type, INDEX_TYPE_BTREE);
        assert_eq!(parsed.key_column, "id");
        assert_eq!(parsed.files.len(), 2);
        assert_eq!(parsed.file_id("/tmp/b.parquet"), Some(1));
        assert_eq!(parsed.file_path(0), Some("/tmp/a.parquet"));
        assert_eq!(parsed.file_path(99), None);
    }
}
