// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `search_keys()` implementation for [`super::ExternalBtreeIndex`].
//!
//! Pipeline:
//!
//! 1. One `BTreeIndex::search(SargableQuery::IsIn(keys))` returns the SET of row
//!    addresses (a `RowAddrTreeMap`) matching any of the keys. `IsIn` gives the
//!    union across keys, not per-key association — sufficient for the
//!    `findTouchedFiles`-style "which target rows match some source key" question.
//! 2. Decode each rid → `(file_id = rid >> 32, row = rid & 0xFFFF_FFFF)` and map
//!    `file_id` → `file_path` via the manifest (same decode as the vector index).
//! 3. Apply the optional [`RowFilter`] (e.g. Delta deletion vectors); rejected
//!    rows are dropped.
//! 4. Return `SearchResult { file_path, row_index, distance: 0.0 }` — `distance`
//!    is meaningless for a scalar point lookup, so it is fixed at 0.0.

use std::sync::Arc;

use datafusion::scalar::ScalarValue;
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::{SargableQuery, ScalarIndex};

use super::manifest::ScalarExternalManifest;
use super::types::{RowFilter, SearchResult};

pub(super) async fn search_keys(
    index: &Arc<dyn ScalarIndex>,
    manifest: &ScalarExternalManifest,
    keys: &[ScalarValue],
    filter: Option<&dyn RowFilter>,
) -> Result<Vec<SearchResult>> {
    if keys.is_empty() {
        return Ok(Vec::new());
    }

    let query = SargableQuery::IsIn(keys.to_vec());
    let result = index.search(&query, &NoOpMetricsCollector).await?;

    // The matched set of row addresses. `true_rows` drops NULL-marked rows (a
    // scalar key lookup never matches NULL). `row_addrs()` is `None` only if the
    // set contains whole-fragment selections, which point lookups never produce.
    let true_rows = result.row_addrs().true_rows();
    let addrs = true_rows.row_addrs().ok_or_else(|| {
        Error::index("btree search returned unbounded (whole-fragment) row addresses")
    })?;

    let mut out = Vec::new();
    for addr in addrs {
        let rid: u64 = addr.into();
        let file_id = (rid >> 32) as u32;
        let row_index = rid & 0xFFFF_FFFF;
        let file_path = manifest
            .file_path(file_id)
            .ok_or_else(|| {
                Error::index(format!(
                    "row addr {rid:#x} encodes file_id={file_id} but manifest has only {} files",
                    manifest.files.len()
                ))
            })?
            .to_string();
        if let Some(f) = filter
            && !f.keep(&file_path, row_index)
        {
            continue;
        }
        out.push(SearchResult {
            file_path,
            row_index,
            distance: 0.0,
        });
    }
    Ok(out)
}
