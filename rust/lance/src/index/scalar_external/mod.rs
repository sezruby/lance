// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! External scalar index — a BTree over caller-supplied parquet files.
//!
//! The scalar analog of [`crate::index::vector::external::ExternalIvfPqIndex`]:
//! Lance builds and queries a BTree over parquet data without copying it into a
//! Lance dataset. The caller registers a list of parquet files; Lance encodes
//! each row's identity as `(file_id_u32 << 32) | row_index_u32` internally, and
//! surfaces lookups as `(file_path, row_index)` so callers never decode the rid
//! themselves.
//!
//! # Why
//!
//! It powers an "indexed Delta MERGE": for each source merge key, find the
//! matching target rows via a BTree over the target's merge key — over Delta's
//! parquet files in place, with no ingestion into a Lance dataset. The returned
//! `(file_path, row_index)` SET is exactly what Delta `findTouchedFiles` needs.
//!
//! # Surface
//!
//! - [`ExternalBtreeIndex`] — handle: build / open / search_keys / fetch_rows
//! - [`ParquetFileSpec`] — describes one parquet file in the registry
//! - [`SearchResult`] — `{file_path, row_index, distance}` (`distance` is 0.0)
//! - [`ParquetRowKey`] — `(file_path, row_index)` accepted by `fetch_rows`
//! - [`RowFilter`] — Delta deletion vectors / Iceberg position deletes / skip
//!   predicates
//! - [`ExternalBtreeIndexParams`] — BTree page-size configuration

mod build;
mod manifest;
mod open;
pub mod params;
mod parquet_source;
mod search;
pub mod types;

pub use params::ExternalBtreeIndexParams;
pub use types::{ParquetFileSpec, ParquetRowKey, PredicateRowFilter, RowFilter, SearchResult};

use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::scalar::ScalarValue;
use lance_core::Result;
use lance_index::scalar::ScalarIndex;
use uuid::Uuid;

use crate::index::vector::external::parquet_source::ParquetMetaCache;
use manifest::ScalarExternalManifest;

/// BTree index over caller-registered parquet files, keyed on a scalar column.
///
/// # Example
///
/// ```ignore
/// # use lance::index::scalar_external::*;
/// # use datafusion::scalar::ScalarValue;
/// # async fn run(files: Vec<ParquetFileSpec>) -> lance_core::Result<()> {
/// let uuid = ExternalBtreeIndex::build(
///     files, "id", "/tmp/idx", ExternalBtreeIndexParams::default(),
/// ).await?;
/// let idx = ExternalBtreeIndex::open(&format!("/tmp/idx/{uuid}")).await?;
/// let hits = idx
///     .search_keys(&[ScalarValue::Int64(Some(42))], None)
///     .await?;
/// for hit in &hits {
///     println!("{} @ row {}", hit.file_path, hit.row_index);
/// }
/// # Ok(()) }
/// ```
pub struct ExternalBtreeIndex {
    /// The loaded BTree, decoupled from any dataset.
    index: Arc<dyn ScalarIndex>,
    /// Parquet file list + key column; provides the `file_id` ↔ `file_path`
    /// mapping used to decode rids returned by `search_keys`.
    manifest: ScalarExternalManifest,
    /// Per-source-parquet metadata cache shared by `fetch_rows`. Populated lazily
    /// on first read of each file; reused across calls on this handle.
    parquet_meta_cache: Arc<ParquetMetaCache>,
}

impl ExternalBtreeIndex {
    /// Build an external BTree index over the given parquet files, keyed on
    /// `key_column`. Writes `<output_uri>/<uuid>/{btree files, manifest.json}`
    /// and returns the `uuid`.
    ///
    /// `key_column` must be a scalar (non-nested) column present in every file's
    /// schema with the same Arrow type. `file_id` is implicit in `files`'s
    /// position — reordering invalidates the index.
    pub async fn build(
        files: Vec<ParquetFileSpec>,
        key_column: &str,
        output_uri: &str,
        params: ExternalBtreeIndexParams,
    ) -> Result<Uuid> {
        build::build_index(files, key_column, output_uri, params).await
    }

    /// Open an external BTree index by URI (the `<output_uri>/<uuid>` directory
    /// `build` wrote). Reads the manifest and loads the BTree lookup metadata.
    pub async fn open(uri: &str) -> Result<Self> {
        open::open_index(uri).await
    }

    /// Number of registered parquet files.
    pub fn num_files(&self) -> usize {
        self.manifest.files.len()
    }

    /// Scalar key column the index was built over.
    pub fn key_column(&self) -> &str {
        &self.manifest.key_column
    }

    /// File path for a given `file_id` (the high 32 bits of the rid). `None` if
    /// the id is out of range.
    pub fn file_path(&self, file_id: u32) -> Option<&str> {
        self.manifest.file_path(file_id)
    }

    /// Look up all rows whose key equals one of `keys`.
    ///
    /// Runs a single BTree `IsIn` query and returns the matched-target-identity
    /// SET as `(file_path, row_index)` (with `distance = 0.0`). `IsIn` yields the
    /// union across keys, not per-key association.
    ///
    /// `filter` (if `Some`) is consulted before a row is emitted; rows it rejects
    /// are dropped. See [`RowFilter`] for typical use cases like Delta deletion
    /// vectors.
    pub async fn search_keys(
        &self,
        keys: &[ScalarValue],
        filter: Option<&dyn RowFilter>,
    ) -> Result<Vec<SearchResult>> {
        search::search_keys(&self.index, &self.manifest, keys, filter).await
    }

    /// Random-access fetch by `(file_path, row_index)` keys.
    ///
    /// Reuses the external vector index's materialization path: Lance batches by
    /// file, issues one page-index-aware parquet read per file, and reassembles
    /// the result in caller-input order (one row per input key). `projection` may
    /// include any parquet column, not just the key column.
    pub async fn fetch_rows(
        &self,
        row_keys: &[ParquetRowKey],
        projection: &[&str],
    ) -> Result<RecordBatch> {
        let first_file_path = self
            .manifest
            .files
            .first()
            .map(|f| f.file_path.as_str())
            .unwrap_or("");
        crate::index::vector::external::fetch::fetch_rows_impl(
            &self.parquet_meta_cache,
            |p| self.manifest.file_id(p).is_some(),
            first_file_path,
            row_keys,
            projection,
        )
        .await
    }
}
