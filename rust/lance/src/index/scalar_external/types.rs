// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Public value types for the external scalar (BTree) index API.
//!
//! These are the same value types the external *vector* index uses — the row
//! identity model (`(file_path, row_index)`, encoded rid `(file_id << 32) | row`),
//! the parquet file registry, the refinement/liveness [`RowFilter`], and the
//! materialization result are all identical across the two indexes. Rather than
//! duplicate them, the scalar module re-exports them so a caller can freely mix a
//! vector and a scalar external index over the same parquet files.

pub use crate::index::vector::external::types::{
    ParquetFileSpec, ParquetRowKey, PredicateRowFilter, RowFilter, SearchResult,
};
