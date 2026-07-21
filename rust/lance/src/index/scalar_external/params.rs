// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build parameters for [`super::ExternalBtreeIndex`].

/// Default BTree page size (rows per page). Mirrors Lance's dataset-backed BTree
/// default and is the value `train_btree_index` chunks the sorted stream into.
pub const DEFAULT_BTREE_BATCH_SIZE: u64 = 4096;

/// Configuration for [`super::ExternalBtreeIndex::build`].
///
/// The key column and file list are passed to `build` directly, so this only
/// carries index-shape knobs. Kept deliberately small — a BTree over a scalar key
/// has far fewer tunables than IVF-PQ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalBtreeIndexParams {
    /// Rows per BTree page. Larger pages mean fewer, coarser pages (cheaper
    /// lookup metadata, more per-page scan); smaller pages mean finer pruning.
    pub batch_size: u64,
}

impl Default for ExternalBtreeIndexParams {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BTREE_BATCH_SIZE,
        }
    }
}

impl ExternalBtreeIndexParams {
    /// Parameters with defaults. Equivalent to [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the BTree page size.
    pub fn with_batch_size(mut self, batch_size: u64) -> Self {
        self.batch_size = batch_size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_override() {
        assert_eq!(
            ExternalBtreeIndexParams::new().batch_size,
            DEFAULT_BTREE_BATCH_SIZE
        );
        assert_eq!(
            ExternalBtreeIndexParams::default()
                .with_batch_size(64)
                .batch_size,
            64
        );
    }
}
