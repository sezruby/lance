// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Public value types and traits for the external vector index API.

use std::sync::Arc;

use arrow_schema::SchemaRef;

/// One parquet file registered with the external index.
///
/// `file_id` for indexed rows is implicit in this struct's position in the
/// `Vec<ParquetFileSpec>` passed to [`super::ExternalIvfPqIndex::build`]. The
/// encoded rid is `(file_id_u32 << 32) | row_index_u32`. Reordering the file list
/// across rebuilds invalidates the index.
#[derive(Clone, Debug)]
pub struct ParquetFileSpec {
    /// URI / path readable by Lance's object store layer.
    pub file_path: String,
    /// Row count, inferred from the parquet footer if not supplied.
    pub num_rows: u64,
    /// Arrow schema of the file. Inferred from the parquet footer if `None` at
    /// build time; populated after the index opens the file.
    pub schema: Option<SchemaRef>,
}

impl ParquetFileSpec {
    /// Construct a spec by inferring `num_rows` and `schema` from the parquet
    /// footer at `file_path`. The caller pays one footer read per file.
    pub fn of(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            num_rows: 0,
            schema: None,
        }
    }

    /// Construct a spec with metadata already known. Skips the footer read.
    pub fn with_metadata(file_path: impl Into<String>, num_rows: u64, schema: SchemaRef) -> Self {
        Self {
            file_path: file_path.into(),
            num_rows,
            schema: Some(schema),
        }
    }
}

/// One result from [`super::ExternalIvfPqIndex::search`]. Already refined; the
/// `distance` is exact under the index's metric.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub file_path: String,
    pub row_index: u64,
    pub distance: f32,
}

/// `(file_path, row_index)` pair accepted by
/// [`super::ExternalIvfPqIndex::fetch_rows`]. Just a value struct — Lance does the
/// per-file batching internally.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParquetRowKey {
    pub file_path: String,
    pub row_index: u64,
}

impl ParquetRowKey {
    pub fn of(file_path: impl Into<String>, row_index: u64) -> Self {
        Self {
            file_path: file_path.into(),
            row_index,
        }
    }
}

/// Filter consulted during refinement. Rows the filter rejects don't make it into
/// the candidate set, so they can't appear in the top-K.
///
/// The intended use case is honoring engine-level row liveness:
///
/// - **Delta deletion vectors**: snapshot's deletion bitmap → `keep` returns false
///   for deleted positions
/// - **Iceberg position deletes**: same shape, different source
/// - **Ad-hoc skip predicates**: e.g. caller already has a snapshot ID and wants to
///   exclude rows newer than it
///
/// Implementations must be `Send + Sync` because Lance may consult them concurrently
/// from refinement workers.
pub trait RowFilter: Send + Sync {
    /// `true` to keep the row, `false` to drop it.
    fn keep(&self, file_path: &str, row_index: u64) -> bool;
}

/// Trivial bitmap-style filter. Wraps a closure.
pub struct PredicateRowFilter<F>(pub F);

impl<F> RowFilter for PredicateRowFilter<F>
where
    F: Fn(&str, u64) -> bool + Send + Sync,
{
    fn keep(&self, file_path: &str, row_index: u64) -> bool {
        (self.0)(file_path, row_index)
    }
}

impl RowFilter for Arc<dyn RowFilter> {
    fn keep(&self, file_path: &str, row_index: u64) -> bool {
        (**self).keep(file_path, row_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rid_encoding_round_trips() {
        // Sanity: confirm the encoding contract Phase 1.3 will rely on.
        let file_id: u32 = 7;
        let row_index: u32 = 42_000;
        let rid = ((file_id as u64) << 32) | (row_index as u64);
        assert_eq!((rid >> 32) as u32, file_id);
        assert_eq!((rid & 0xFFFF_FFFF) as u32, row_index);
    }

    #[test]
    fn predicate_row_filter_keeps_and_drops() {
        let f = PredicateRowFilter(|_path: &str, row: u64| row % 2 == 0);
        assert!(f.keep("a.parquet", 0));
        assert!(!f.keep("a.parquet", 1));
        assert!(f.keep("b.parquet", 100));
    }

    #[test]
    fn parquet_file_spec_constructors() {
        let s = ParquetFileSpec::of("/tmp/x.parquet");
        assert_eq!(s.file_path, "/tmp/x.parquet");
        assert_eq!(s.num_rows, 0);
        assert!(s.schema.is_none());
    }
}
