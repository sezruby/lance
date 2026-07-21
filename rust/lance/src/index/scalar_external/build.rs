// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `build()` implementation for [`super::ExternalBtreeIndex`].
//!
//! Composes the existing public Lance scalar-index APIs, with the `Dataset`
//! dependency removed end-to-end:
//!
//! 1. [`sorted_value_rowid_stream`] reads the key column from each parquet file,
//!    attaches the rid, renames the key to `"value"`, unions all files, and sorts
//!    ascending by `value`.
//! 2. `train_btree_index` consumes that sorted stream, writing the BTree lookup +
//!    page files into a [`LanceIndexStore`] rooted at `<output_uri>/<uuid>/`.
//! 3. A scalar-flavored `manifest.json` (file list + key column + index-type tag)
//!    is written alongside so `open()` can decode rids back to `(file_path, row)`.
//!
//! Layout written under `output_uri`:
//!
//! ```text
//! <output_uri>/
//!   <index_uuid>/
//!     btree_lookup.lance  ← BTree page-lookup metadata
//!     btree_pages.lance   ← BTree pages (sorted key/rid pairs)
//!     manifest.json       ← ParquetFileSpec list + key column + index type
//! ```

use std::sync::Arc;

use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use lance_index::scalar::btree::train_btree_index;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_io::object_store::ObjectStore;
use uuid::Uuid;

use super::manifest::{ScalarExternalManifest, write_manifest};
use super::params::ExternalBtreeIndexParams;
use super::parquet_source::{read_num_rows, sorted_value_rowid_stream};
use super::types::ParquetFileSpec;

pub(super) async fn build_index(
    files: Vec<ParquetFileSpec>,
    key_column: &str,
    output_uri: &str,
    params: ExternalBtreeIndexParams,
) -> Result<Uuid> {
    if files.is_empty() {
        return Err(Error::invalid_input(
            "ExternalBtreeIndex requires at least one parquet file",
        ));
    }

    // Resolve output_uri → (ObjectStore, Path); the index files + manifest live
    // under <output_uri>/<uuid>/.
    let (object_store, root_path) = ObjectStore::from_uri(output_uri).await?;
    let index_uuid = Uuid::new_v4();
    let index_dir = root_path.clone().join(index_uuid.to_string());
    let store = Arc::new(LanceIndexStore::new(
        object_store.clone(),
        index_dir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    // Train the BTree over the sorted (value, _rowid) stream. This is the only
    // step that needs a corpus-wide view; it streams from the source parquet.
    let stream = sorted_value_rowid_stream(&files, key_column).await?;
    train_btree_index(stream, store.as_ref(), params.batch_size, None, None).await?;

    // Resolve any unfilled num_rows from the footers so the manifest records
    // authoritative counts (used for rid decode bounds and debugging).
    let mut resolved_files = files;
    for spec in resolved_files.iter_mut() {
        if spec.num_rows == 0 {
            spec.num_rows = read_num_rows(&spec.file_path).await?;
        }
    }

    let manifest = ScalarExternalManifest::from_build(key_column, &resolved_files);
    write_manifest(&object_store, &index_dir, &manifest).await?;

    Ok(index_uuid)
}
