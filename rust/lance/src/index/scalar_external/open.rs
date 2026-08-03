// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `open()` implementation for [`super::ExternalBtreeIndex`].
//!
//! Reads `<index_dir>/manifest.json` (parquet file list + key column) and loads
//! the BTree via the public [`BTreeIndexPlugin`], which returns an
//! `Arc<dyn ScalarIndex>` over the same [`LanceIndexStore`] the build wrote. No
//! `Dataset` is involved — the index is self-contained on disk.

use std::sync::Arc;

use lance_core::Result;
use lance_core::cache::LanceCache;
use lance_index::scalar::IndexStore;
use lance_index::scalar::btree::BTreeIndexPlugin;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_io::object_store::ObjectStore;

use super::manifest::read_manifest;
use crate::index::vector::external::parquet_source::ParquetMetaCache;

pub(super) async fn open_index(uri: &str) -> Result<super::ExternalBtreeIndex> {
    let (object_store, index_dir) = ObjectStore::from_uri(uri).await?;

    let manifest = read_manifest(&object_store, &index_dir).await?;

    let cache = Arc::new(LanceCache::no_cache());
    let store: Arc<dyn IndexStore> =
        Arc::new(LanceIndexStore::new(object_store, index_dir, cache.clone()));

    // The BTree plugin ignores index_details on load, so a default `Any` is fine.
    let index = BTreeIndexPlugin
        .load_index(store, &prost_types::Any::default(), None, cache.as_ref())
        .await?;

    Ok(super::ExternalBtreeIndex {
        index,
        manifest,
        parquet_meta_cache: Arc::new(ParquetMetaCache::new()),
    })
}
