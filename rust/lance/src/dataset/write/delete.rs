// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::rowids::get_row_id_index;
use crate::dataset::scanner::ExprFilter;
use crate::{
    Dataset,
    dataset::transaction::{Operation, Transaction},
    dataset::utils::make_rowid_capture_stream,
};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, ROW_ID, Result};
use lance_select::RowAddrTreeMap;
use lance_table::format::Fragment;
use roaring::RoaringTreemap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use super::CommitBuilder;
use super::retry::{RetryConfig, RetryExecutor, execute_with_retry};

/// Result of a delete operation.
#[derive(Debug, Clone)]
pub struct DeleteResult {
    /// The new dataset after the delete operation.
    pub new_dataset: Arc<Dataset>,
    /// The number of rows that were deleted.
    pub num_deleted_rows: u64,
}

/// Result of a staged delete operation.
///
/// The returned transaction can be committed later with [`CommitBuilder`].
/// Pass `affected_rows` to [`CommitBuilder::with_affected_rows`] when present
/// to preserve row-level conflict resolution for concurrent deletes and updates.
#[derive(Debug, Clone)]
pub struct UncommittedDelete {
    /// The transaction to commit.
    pub transaction: Transaction,
    /// The row addresses affected by the delete, if available.
    pub affected_rows: Option<RowAddrTreeMap>,
    /// The number of rows that were deleted.
    pub num_deleted_rows: u64,
}

/// Apply deletions to fragments based on a RoaringTreemap of row IDs.
///
/// Returns the set of modified fragments and removed fragments, if any.
async fn apply_deletions(
    dataset: &Dataset,
    removed_row_addrs: &RoaringTreemap,
) -> Result<(Vec<Fragment>, Vec<u64>)> {
    let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

    enum FragmentChange {
        Unchanged,
        Modified(Box<Fragment>),
        Removed(u64),
    }

    let mut updated_fragments = Vec::new();
    let mut removed_fragments = Vec::new();

    let mut stream = futures::stream::iter(dataset.get_fragments())
        .map(move |fragment| {
            let bitmaps_ref = bitmaps.clone();
            async move {
                let fragment_id = fragment.id();
                if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                    match fragment.extend_deletions(*bitmap).await {
                        Ok(Some(new_fragment)) => {
                            Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
                        }
                        Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                        Err(e) => Err(e),
                    }
                } else {
                    Ok(FragmentChange::Unchanged)
                }
            }
        })
        .buffer_unordered(dataset.object_store.io_parallelism());

    while let Some(res) = stream.next().await.transpose()? {
        match res {
            FragmentChange::Unchanged => {}
            FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
            FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
        }
    }

    Ok((updated_fragments, removed_fragments))
}

/// Builder for configuring delete operations with retry support
///
/// This operation is similar to SQL's DELETE statement. It allows you to remove
/// rows from a dataset based on a filter predicate with automatic retry support
/// for handling concurrent write conflicts.
///
/// Use the [DeleteBuilder] to construct a delete operation. For example:
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::DeleteBuilder;
/// # use std::sync::Arc;
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let result = DeleteBuilder::new(dataset, "age > 65")
///     .conflict_retries(5)
///     .execute()
///     .await?;
/// println!("Deleted {} rows", result.num_deleted_rows);
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct DeleteBuilder {
    dataset: Arc<Dataset>,
    filter: ExprFilter,
    conflict_retries: u32,
    retry_timeout: Duration,
    target_fragments: Option<Vec<u32>>,
}

impl DeleteBuilder {
    /// Create a new DeleteBuilder with a SQL predicate string
    pub fn new(dataset: Arc<Dataset>, predicate: impl Into<String>) -> Self {
        Self {
            dataset,
            filter: ExprFilter::Sql(predicate.into()),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
            target_fragments: None,
        }
    }

    /// Create a new DeleteBuilder with a DataFusion expression filter
    pub fn from_expr(dataset: Arc<Dataset>, expr: Expr) -> Self {
        Self {
            dataset,
            filter: ExprFilter::Datafusion(expr),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
            target_fragments: None,
        }
    }

    /// Set the number of retries for conflict resolution
    pub fn conflict_retries(mut self, retries: u32) -> Self {
        self.conflict_retries = retries;
        self
    }

    /// Set the timeout for retry operations
    pub fn retry_timeout(mut self, timeout: Duration) -> Self {
        self.retry_timeout = timeout;
        self
    }

    /// Restrict the delete to a subset of the dataset's fragments.
    ///
    /// The predicate is evaluated only against rows living in the given
    /// fragments, and the resulting transaction only tombstones rows in those
    /// fragments. This is the producer side of a distributed / parallel delete:
    /// partition the target's fragments into disjoint slices, run one
    /// [`Self::execute_uncommitted`] per slice (in parallel), then commit all of
    /// the resulting transactions together with
    /// [`CommitBuilder::execute_batch`]. Because the slices are disjoint, no two
    /// tasks can tombstone the same physical row, so the batch commit is
    /// conflict-free and equivalent to a single full delete with the same
    /// predicate over the whole dataset.
    ///
    /// Unknown fragment ids are rejected when the delete executes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lance::dataset::DeleteBuilder;
    ///
    /// # use std::sync::Arc;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let staged_delete = DeleteBuilder::new(dataset, "age > 65")
    ///     .with_target_fragments(vec![0, 1])
    ///     .execute_uncommitted()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_target_fragments(mut self, fragment_ids: Vec<u32>) -> Self {
        self.target_fragments = Some(fragment_ids);
        self
    }

    /// Execute the delete operation
    pub async fn execute(self) -> Result<DeleteResult> {
        let job = DeleteJob {
            dataset: self.dataset.clone(),
            filter: self.filter,
            target_fragments: self.target_fragments,
        };

        let config = RetryConfig {
            max_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        };

        execute_with_retry(job, self.dataset, config).await
    }

    /// Execute the delete operation without committing the transaction.
    ///
    /// Use [`CommitBuilder`] to commit the returned transaction.
    ///
    /// # Example: Delete rows from a dataset
    ///
    /// ```rust
    /// use lance::dataset::{CommitBuilder, DeleteBuilder};
    ///
    /// # use std::sync::Arc;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let staged_delete = DeleteBuilder::new(dataset.clone(), "age > 65")
    ///     .execute_uncommitted()
    ///     .await?;
    /// let mut commit_builder = CommitBuilder::new(dataset);
    /// if let Some(affected_rows) = staged_delete.affected_rows {
    ///     commit_builder = commit_builder.with_affected_rows(affected_rows);
    /// }
    /// commit_builder
    ///     .execute(staged_delete.transaction)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted(self) -> Result<UncommittedDelete> {
        let job = DeleteJob {
            dataset: self.dataset,
            filter: self.filter,
            target_fragments: self.target_fragments,
        };
        let data = job.execute_impl().await?;
        let DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        } = data;
        let transaction = job.build_transaction(
            job.dataset.as_ref(),
            updated_fragments,
            deleted_fragment_ids,
        );
        Ok(UncommittedDelete {
            transaction,
            affected_rows,
            num_deleted_rows,
        })
    }
}

/// Job that executes the delete operation
#[derive(Debug, Clone)]
struct DeleteJob {
    dataset: Arc<Dataset>,
    filter: ExprFilter,
    /// When set, restrict the delete to this subset of the dataset's fragment ids.
    target_fragments: Option<Vec<u32>>,
}

/// Data returned by delete operation
struct DeleteData {
    updated_fragments: Vec<Fragment>,
    deleted_fragment_ids: Vec<u64>,
    affected_rows: Option<RowAddrTreeMap>,
    num_deleted_rows: u64,
}

impl DeleteJob {
    /// Resolve `target_fragments` ids into the dataset's [`Fragment`] metadata,
    /// erroring on any id that does not exist or is repeated. Returns `None` for
    /// an unscoped (whole-dataset) delete.
    fn resolve_target_fragments(&self) -> Result<Option<Vec<Fragment>>> {
        let Some(fragment_ids) = &self.target_fragments else {
            return Ok(None);
        };
        let by_id: BTreeMap<u64, &Fragment> =
            self.dataset.fragments().iter().map(|f| (f.id, f)).collect();
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let fragments = fragment_ids
            .iter()
            .map(|id| {
                if !seen.insert(*id) {
                    return Err(Error::invalid_input(format!(
                        "target_fragments contains duplicate fragment id {id}"
                    )));
                }
                by_id.get(&(*id as u64)).map(|f| (*f).clone()).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "target_fragments references fragment id {} which does not exist in the dataset",
                        id
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(fragments))
    }

    fn build_transaction(
        &self,
        dataset: &Dataset,
        updated_fragments: Vec<Fragment>,
        deleted_fragment_ids: Vec<u64>,
    ) -> Transaction {
        let predicate = match &self.filter {
            ExprFilter::Sql(s) => s.clone(),
            ExprFilter::Datafusion(expr) => expr.to_string(),
            ExprFilter::Substrait(_) => {
                unreachable!("Substrait filters are not supported in DeleteBuilder")
            }
        };
        let operation = Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        };
        Transaction::new(dataset.manifest.version, operation, None)
    }
}

impl RetryExecutor for DeleteJob {
    type Data = DeleteData;
    type Result = DeleteResult;

    async fn execute_impl(&self) -> Result<Self::Data> {
        // Resolve the fragment slice up front (if any). A scoped delete only reads
        // and tombstones rows in these fragments, so the resulting transaction is
        // disjoint from any other slice's — enabling a distributed delete to be
        // batch-committed conflict-free (see `combine_delete_transactions`).
        let target_fragments = self.resolve_target_fragments()?;

        // Create a scanner over the whole dataset, or just the target slice.
        let mut scanner = self.dataset.scan();
        scanner.with_row_id().project(&[ROW_ID])?;
        if let Some(fragments) = &target_fragments {
            scanner.with_fragments(fragments.clone());
        }
        match &self.filter {
            ExprFilter::Sql(s) => {
                scanner.filter(s)?;
            }
            ExprFilter::Datafusion(expr) => {
                scanner.filter_expr(expr.clone());
            }
            ExprFilter::Substrait(_) => {
                unreachable!("Substrait filters are not supported in DeleteBuilder")
            }
        }

        // Check if the filter optimized to true (delete everything) or false (delete nothing)
        let (updated_fragments, deleted_fragment_ids, affected_rows, num_deleted_rows) =
            if let Some(filter_expr) = scanner.get_expr_filter()? {
                if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(false)), _)
                ) {
                    // Predicate evaluated to false - no deletions
                    (Vec::new(), Vec::new(), Some(RowAddrTreeMap::new()), 0)
                } else if matches!(
                    filter_expr,
                    Expr::Literal(ScalarValue::Boolean(Some(true)), _)
                ) {
                    // Predicate evaluated to true - delete every targeted fragment.
                    // When scoped, only the slice's fragments are removed so the
                    // transaction stays disjoint from other slices.
                    let fragments = match &target_fragments {
                        Some(fragments) => fragments.clone(),
                        None => self
                            .dataset
                            .get_fragments()
                            .into_iter()
                            .map(|f| f.metadata)
                            .collect(),
                    };
                    let num_deleted_rows: u64 = fragments
                        .iter()
                        .map(|f| f.num_rows().unwrap_or(0) as u64)
                        .sum();
                    let deleted_fragment_ids = fragments.iter().map(|f| f.id).collect();

                    // When deleting everything, we don't have specific row addresses,
                    // so better not to emit affected rows.
                    (Vec::new(), deleted_fragment_ids, None, num_deleted_rows)
                } else {
                    // Regular predicate - scan and collect row addresses to delete
                    let stream = scanner.try_into_stream().await?.into();
                    let (stream, row_id_rx) = make_rowid_capture_stream(
                        stream,
                        self.dataset.manifest.uses_stable_row_ids(),
                    )?;

                    // Process the stream to capture row addresses
                    // We need to consume the stream to trigger the capture
                    futures::pin_mut!(stream);
                    while let Some(_batch) = stream.try_next().await? {
                        // The row addresses are captured automatically by make_rowid_capture_stream
                    }

                    // Extract the row addresses from the receiver
                    let removed_row_ids = row_id_rx.try_recv().map_err(|err| {
                        Error::internal(format!("Failed to receive row ids: {}", err))
                    })?;
                    let row_id_index = get_row_id_index(&self.dataset).await?;
                    let removed_row_addrs = removed_row_ids.row_addrs(row_id_index.as_deref());

                    let (fragments, deleted_ids) =
                        apply_deletions(&self.dataset, &removed_row_addrs).await?;
                    let num_deleted_rows = removed_row_addrs.len();
                    let affected_rows = RowAddrTreeMap::from(removed_row_addrs.as_ref().clone());
                    (
                        fragments,
                        deleted_ids,
                        Some(affected_rows),
                        num_deleted_rows,
                    )
                }
            } else {
                // No filter was applied - this shouldn't happen but treat as delete nothing
                (Vec::new(), Vec::new(), Some(RowAddrTreeMap::new()), 0)
            };

        Ok(DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        })
    }

    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result> {
        let DeleteData {
            updated_fragments,
            deleted_fragment_ids,
            affected_rows,
            num_deleted_rows,
        } = data;
        let transaction =
            self.build_transaction(dataset.as_ref(), updated_fragments, deleted_fragment_ids);

        let mut builder = CommitBuilder::new(dataset);

        if let Some(affected_rows) = affected_rows {
            builder = builder.with_affected_rows(affected_rows);
        }

        let new_dataset = builder.execute(transaction).await.map(Arc::new)?;
        Ok(DeleteResult {
            new_dataset,
            num_deleted_rows,
        })
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

/// Legacy delete function - uses DeleteBuilder with no retries for backwards compatibility
pub async fn delete(ds: &mut Dataset, predicate: &str) -> Result<DeleteResult> {
    // Use DeleteBuilder with 0 retries to maintain backwards compatibility
    let dataset = Arc::new(ds.clone());
    let result = DeleteBuilder::new(dataset, predicate).execute().await?;

    // Update the dataset in place
    *ds = Arc::try_unwrap(result.new_dataset.clone()).unwrap_or_else(|arc| (*arc).clone());
    Ok(result)
}

/// A combined delete produced by [`combine_delete_transactions`], ready to
/// commit with [`CommitBuilder`].
#[derive(Debug, Clone)]
pub struct CombinedDelete {
    /// The single [`Operation::Delete`] transaction covering every slice.
    pub transaction: Transaction,
    /// The row addresses newly tombstoned across all slices, relative to the
    /// shared `read_version`. Pass to [`CommitBuilder::with_affected_rows`] so a
    /// concurrent writer can be rebased against at row granularity — matching a
    /// single [`DeleteBuilder::execute`].
    pub affected_rows: RowAddrTreeMap,
    /// Total number of rows deleted across all slices.
    pub num_deleted_rows: u64,
}

/// Combine several fragment-scoped [`Operation::Delete`] transactions into a
/// single delete transaction that can be committed once.
///
/// This is the driver-side step of a distributed / parallel delete: each task
/// produced an independent [`UncommittedDelete`] over a **disjoint** slice of
/// the dataset's fragments (via [`DeleteBuilder::with_target_fragments`] +
/// [`DeleteBuilder::execute_uncommitted`]). This function stitches their
/// [`Operation::Delete`] metadata into one transaction — it never reads or
/// rewrites data files, and never rewrites deletion files (each task already
/// wrote its own fragments' deletion files):
///
/// - `updated_fragments` from every transaction are concatenated.
/// - `deleted_fragment_ids` (fragments deleted in whole) are concatenated.
/// - All input transactions must be [`Operation::Delete`], share the same
///   `read_version`, and use the same `predicate`; otherwise an error is
///   returned.
///
/// # Disjointness
///
/// The slices must be disjoint, so each fragment id is modified by at most one
/// transaction and the combined result is exactly equal to a single full delete
/// of the same predicate over the whole dataset. If any fragment id appears in
/// more than one transaction (only possible if the caller passed overlapping
/// slices), this errors rather than silently committing — covering both
/// partially-updated fragments and whole-fragment deletions.
///
/// To rebase safely against concurrent writers, [`CombinedDelete::affected_rows`]
/// is reconstructed by diffing each touched fragment's post-delete deletion
/// vector against its state at `read_version`.
pub async fn combine_delete_transactions(
    dataset: &Dataset,
    transactions: Vec<Transaction>,
) -> Result<CombinedDelete> {
    if transactions.is_empty() {
        return Err(Error::invalid_input(
            "combine_delete_transactions requires at least one transaction".to_string(),
        ));
    }

    let read_version = transactions[0].read_version;
    let mut combined_predicate: Option<String> = None;
    let mut updated_fragments: Vec<Fragment> = Vec::new();
    let mut deleted_fragment_ids: Vec<u64> = Vec::new();
    // Every fragment id seen so far, to enforce disjointness across slices. A
    // fragment appearing twice (whether updated or wholly deleted) means the
    // caller passed overlapping slices.
    let mut seen_fragment_ids: BTreeMap<u64, ()> = BTreeMap::new();

    for txn in &transactions {
        if txn.read_version != read_version {
            return Err(Error::invalid_input(format!(
                "all delete transactions must share read_version; expected {}, got {}",
                read_version, txn.read_version
            )));
        }
        let Operation::Delete {
            updated_fragments: txn_updated,
            deleted_fragment_ids: txn_deleted,
            predicate,
        } = &txn.operation
        else {
            return Err(Error::invalid_input(format!(
                "combine_delete_transactions only supports Operation::Delete, got {}",
                txn.operation.name()
            )));
        };

        match &combined_predicate {
            Some(existing) if existing != predicate => {
                return Err(Error::invalid_input(format!(
                    "all delete transactions must share the same predicate; expected {:?}, got {:?}",
                    existing, predicate
                )));
            }
            Some(_) => {}
            None => combined_predicate = Some(predicate.clone()),
        }

        for id in txn_updated
            .iter()
            .map(|f| f.id)
            .chain(txn_deleted.iter().copied())
        {
            if seen_fragment_ids.insert(id, ()).is_some() {
                return Err(Error::invalid_input(format!(
                    "two delete transactions both modified fragment {id} — the fragment slices \
                     passed to with_target_fragments are not disjoint"
                )));
            }
        }

        updated_fragments.extend(txn_updated.iter().cloned());
        deleted_fragment_ids.extend(txn_deleted.iter().copied());
    }

    // Deterministic ordering for the committed transaction.
    updated_fragments.sort_unstable_by_key(|f| f.id);
    deleted_fragment_ids.sort_unstable();

    // Reconstruct the newly-deleted row addresses (delta vs read_version) so the
    // batch commit can rebase against a concurrent writer at row granularity,
    // just as a single delete does. This reads deletion files only; data files
    // are untouched. `predicate` is guaranteed Some (non-empty input checked
    // above); a missing predicate is a programming error.
    let predicate = combined_predicate
        .ok_or_else(|| Error::internal("combined delete produced no predicate".to_string()))?;
    let (affected_rows, num_deleted_rows) = delete_delta_affected_rows(
        dataset,
        read_version,
        &updated_fragments,
        &deleted_fragment_ids,
    )
    .await?;

    let operation = Operation::Delete {
        updated_fragments,
        deleted_fragment_ids,
        predicate,
    };

    Ok(CombinedDelete {
        transaction: Transaction::new(read_version, operation, None),
        affected_rows,
        num_deleted_rows,
    })
}

/// Compute the row addresses newly tombstoned by a combined delete, relative to
/// `read_version`: for each updated fragment, the post-delete deletion vector
/// minus the deletions that already existed at `read_version`; for each
/// wholly-deleted fragment, all of its still-live rows at `read_version`.
async fn delete_delta_affected_rows(
    dataset: &Dataset,
    read_version: u64,
    updated_fragments: &[Fragment],
    deleted_fragment_ids: &[u64],
) -> Result<(RowAddrTreeMap, u64)> {
    let base = dataset.checkout_version(read_version).await?;
    let base = &base;
    let base_by_id: BTreeMap<u64, &Fragment> = base.fragments().iter().map(|f| (f.id, f)).collect();
    let base_by_id = &base_by_id;
    let io_parallelism = dataset.object_store.io_parallelism();

    // Per-fragment newly-deleted bitmaps, computed concurrently (I/O bound).
    let per_fragment: Vec<(u32, roaring::RoaringBitmap)> = futures::stream::iter(
        updated_fragments
            .iter()
            .map(|f| (f.id, Some(f)))
            .chain(deleted_fragment_ids.iter().map(|id| (*id, None))),
    )
    .map(|(frag_id, updated)| {
        let base_fragment = base_by_id.get(&frag_id).copied();
        async move {
            let before = match base_fragment {
                Some(bf) => read_fragment_deletions(base, bf).await?,
                // Fragment not present at read_version: nothing was deleted before.
                None => roaring::RoaringBitmap::new(),
            };
            let after = match updated {
                // Partial delete: read the fragment's post-delete deletion vector.
                Some(f) => read_fragment_deletions(dataset, f).await?,
                // Whole-fragment delete: every physical row is now tombstoned.
                None => match base_fragment.and_then(|bf| bf.physical_rows) {
                    Some(rows) => (0..rows as u32).collect(),
                    None => roaring::RoaringBitmap::new(),
                },
            };
            Ok::<_, Error>((frag_id as u32, after - before))
        }
    })
    .buffer_unordered(io_parallelism)
    .try_collect()
    .await?;

    let mut affected_rows = RowAddrTreeMap::new();
    let mut num_deleted_rows: u64 = 0;
    for (frag_id, bitmap) in per_fragment {
        num_deleted_rows += bitmap.len();
        if !bitmap.is_empty() {
            affected_rows.insert_bitmap(frag_id, bitmap);
        }
    }
    Ok((affected_rows, num_deleted_rows))
}

/// Read a fragment's deletion vector as a `RoaringBitmap` (empty if none).
async fn read_fragment_deletions(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<roaring::RoaringBitmap> {
    use crate::io::deletion::read_dataset_deletion_file;
    match &fragment.deletion_file {
        Some(df) => {
            let dv = read_dataset_deletion_file(dataset, fragment.id, df).await?;
            Ok(roaring::RoaringBitmap::from_iter(
                dv.as_ref().to_sorted_iter(),
            ))
        }
        None => Ok(roaring::RoaringBitmap::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, UpdateBuilder};
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::TestDatasetGenerator;
    use arrow::array::AsArray;
    use arrow::datatypes::UInt32Type;
    use arrow_array::{RecordBatch, UInt32Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_select::mask::RowSetOps;
    use rstest::rstest;
    use std::collections::HashSet;
    use std::ops::Range;
    use std::sync::Arc;

    #[rstest]
    #[tokio::test]
    async fn test_delete(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] with_scalar_index: bool,
    ) {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }
        // Write a dataset
        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();
        let data = sequence_data(0..100);
        // Split over two files.
        let batches = vec![data.slice(0, 50), data.slice(50, 50)];
        let mut dataset = TestDatasetGenerator::new(batches, data_storage_version)
            .make_hostile(&tmp_path)
            .await;

        if with_scalar_index {
            dataset
                .create_index(
                    &["i"],
                    IndexType::Scalar,
                    Some("scalar_index".to_string()),
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        // Delete nothing
        let result = dataset.delete("i < 0").await.unwrap();
        assert_eq!(result.num_deleted_rows, 0);
        dataset.validate().await.unwrap();

        // We should not have any deletion file still
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 0);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(1));
        assert!(fragments[0].metadata.deletion_file.is_none());
        assert!(fragments[1].metadata.deletion_file.is_none());

        // Delete rows
        let result = dataset.delete("i < 10 OR i >= 90").await.unwrap();
        assert_eq!(result.num_deleted_rows, 20);
        dataset.validate().await.unwrap();

        // Verify result:
        // There should be a deletion file in the metadata
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        assert!(fragments[1].metadata.deletion_file.is_some());
        assert_eq!(
            fragments[0]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows,
            Some(10)
        );

        // The deletion file should contain 20 rows
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);
        // First fragment has 0..10 deleted
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..10).collect::<HashSet<_>>()
        );
        // Second fragment has 90..100 deleted
        let deletion_vector = fragments[1].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 10);
        // The second fragment starts at 50, so 90..100 becomes 40..50 in local row ids.
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (40..50).collect::<HashSet<_>>()
        );
        let second_deletion_file = fragments[1].metadata.deletion_file.clone().unwrap();

        // Delete more rows (only 10 new rows since 0..10 already deleted)
        let result = dataset.delete("i < 20").await.unwrap();
        assert_eq!(result.num_deleted_rows, 10);
        dataset.validate().await.unwrap();

        // Verify result
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 30);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert!(fragments[0].metadata.deletion_file.is_some());
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 20);
        assert_eq!(
            deletion_vector.iter().collect::<HashSet<_>>(),
            (0..20).collect::<HashSet<_>>()
        );
        // Second deletion vector was not rewritten
        assert_eq!(
            fragments[1].metadata.deletion_file.as_ref().unwrap(),
            &second_deletion_file
        );

        // Delete full fragment (50 rows remaining in fragment 1, 10 already deleted)
        let result = dataset.delete("i >= 50").await.unwrap();
        assert_eq!(result.num_deleted_rows, 40);
        dataset.validate().await.unwrap();

        // Verify second fragment is fully gone
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert_eq!(dataset.count_fragments(), 1);
        assert_eq!(fragments[0].id(), 0);

        // Verify the count_deleted_rows only contains the rows from the first fragment
        // i.e. - deleted_rows from the fragment that has been deleted are not counted
        assert_eq!(dataset.count_deleted_rows().await.unwrap(), 20);

        // Append after delete
        let data = sequence_data(0..100);
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap();

        dataset.validate().await.unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 2);
        assert_eq!(dataset.count_fragments(), 2);
        // Fragment id picks up where we left off
        assert_eq!(fragments[0].id(), 0);
        assert_eq!(fragments[1].id(), 2);
        assert_eq!(dataset.manifest.max_fragment_id(), Some(2));
    }

    #[tokio::test]
    async fn test_delete_with_single_scanner() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::UInt32, false),
                ArrowField::new("x", DataType::UInt32, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt32Array::from_iter_values(range.clone())),
                    Arc::new(UInt32Array::from_iter_values(range.map(|v| v * 2))),
                ],
            )
            .unwrap()
        }

        // Create dataset with multiple fragments
        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

        // Create 5 fragments with 100 rows each
        let mut batches = Vec::new();
        for i in 0..5 {
            let start = i * 100;
            let end = (i + 1) * 100;
            let data = sequence_data(start..end);
            batches.push(data);
        }

        let mut dataset = TestDatasetGenerator::new(batches, LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        // Delete rows across multiple fragments using the new scanner-based implementation
        let predicate = "i >= 50 AND i < 150";
        dataset.delete(predicate).await.unwrap();

        // Verify the deletion worked correctly
        let mut scanner = dataset.scan();
        scanner.filter(predicate).unwrap();
        let count = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_fold(0, |acc, batch| async move { Ok(acc + batch.num_rows()) })
            .await
            .unwrap();

        assert_eq!(
            count, 0,
            "All rows matching the predicate should be deleted"
        );

        // Verify that rows outside the predicate still exist
        let mut remaining_scanner = dataset.scan();
        remaining_scanner.filter("i < 50 OR i >= 150").unwrap();
        let remaining_count = remaining_scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_fold(0, |acc, batch| async move { Ok(acc + batch.num_rows()) })
            .await
            .unwrap();

        assert_eq!(
            remaining_count, 400,
            "400 rows should remain after deletion"
        );

        // Check that fragments were handled correctly
        let fragments = dataset.get_fragments();
        assert!(fragments.len() == 5, "All fragments should still exist");

        // Fragment 0 (rows 0-99) should have 50 deletions (50-99)
        let frag0_dv = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(frag0_dv.len(), 50);

        // Fragment 1 (rows 100-199) should be fully deleted or have 50 deletions (100-149)
        let frag1_dv = fragments[1].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(frag1_dv.len(), 50);
    }

    #[tokio::test]
    async fn test_delete_false_predicate_still_commits() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

        let data = sequence_data(0..100);
        let mut dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        let initial_version = dataset.version().version;

        // Delete with false predicate - should still commit but not delete anything
        dataset.delete("false").await.unwrap();

        // Verify version incremented (commit happened)
        assert_eq!(dataset.version().version, initial_version + 1);

        // Verify no rows were deleted
        assert_eq!(dataset.count_rows(None).await.unwrap(), 100);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }

    #[tokio::test]
    async fn test_delete_execute_uncommitted_preserves_affected_rows_for_rebase() {
        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

        let dataset = InsertBuilder::new(&tmp_path)
            .execute(vec![sequence_data(0..100)])
            .await
            .unwrap();
        let initial_version = dataset.version().version;

        let staged_delete = DeleteBuilder::new(Arc::new(dataset.clone()), "i < 10")
            .execute_uncommitted()
            .await
            .unwrap();

        let dataset_before_commit = Dataset::open(&tmp_path).await.unwrap();
        assert_eq!(dataset_before_commit.version().version, initial_version);
        assert_eq!(dataset_before_commit.count_rows(None).await.unwrap(), 100);

        assert_eq!(staged_delete.num_deleted_rows, 10);
        assert!(staged_delete.affected_rows.is_some());
        assert_eq!(staged_delete.transaction.read_version, initial_version);
        match &staged_delete.transaction.operation {
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => {
                assert_eq!(predicate, "i < 10");
                assert_eq!(updated_fragments.len(), 1);
                assert!(deleted_fragment_ids.is_empty());
            }
            other => panic!("expected delete transaction, got {other:?}"),
        }

        DeleteBuilder::new(Arc::new(dataset.clone()), "i >= 10 AND i < 20")
            .execute()
            .await
            .unwrap();

        let mut commit_builder = CommitBuilder::new(&tmp_path);
        if let Some(affected_rows) = staged_delete.affected_rows {
            commit_builder = commit_builder.with_affected_rows(affected_rows);
        }
        let committed = commit_builder
            .execute(staged_delete.transaction)
            .await
            .unwrap();
        assert_eq!(committed.version().version, initial_version + 2);
        assert_eq!(committed.count_rows(None).await.unwrap(), 80);
    }

    #[tokio::test]
    async fn test_concurrent_delete_with_retries() {
        use futures::future::try_join_all;
        use tokio::sync::Barrier;

        fn sequence_data(range: Range<u32>) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "i",
                DataType::UInt32,
                false,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt32Array::from_iter_values(range))])
                .unwrap()
        }

        let tmp_dir = TempStrDir::default();
        let tmp_path = tmp_dir.as_str().to_string();

        let data = sequence_data(0..100);
        let dataset = TestDatasetGenerator::new(vec![data], LanceFileVersion::Stable)
            .make_hostile(&tmp_path)
            .await;

        let concurrency = 3;
        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();

        // Create multiple concurrent delete operations targeting the same overlapping range
        // All tasks try to delete the same set of rows (0-49), creating maximum conflict
        for _i in 0..concurrency {
            let dataset_ref = Arc::new(dataset.clone());
            let barrier_ref = barrier.clone();

            let handle = tokio::spawn(async move {
                barrier_ref.wait().await;

                DeleteBuilder::new(dataset_ref, "i < 50") // All tasks delete the same rows
                    .conflict_retries(5)
                    .execute()
                    .await
            });
            handles.push(handle);
        }

        // All tasks should complete successfully with retry-based conflict resolution
        let results = try_join_all(handles).await.unwrap();

        // All delete operations should succeed
        for result in &results {
            assert!(
                result.is_ok(),
                "Delete operation should succeed with retries"
            );
        }

        // Get the final dataset from any successful result
        let final_result = results.into_iter().find_map(|r| r.ok()).unwrap();
        let final_dataset = final_result.new_dataset;

        // Rows 0-49 should be deleted, rows 50-99 should remain
        assert_eq!(final_dataset.count_rows(None).await.unwrap(), 50);

        // Verify the remaining data is rows 50-99
        let data = final_dataset.scan().try_into_batch().await.unwrap();
        let remaining_values: Vec<u32> = data["i"].as_primitive::<UInt32Type>().values().to_vec();
        let expected: Vec<u32> = (50..100).collect();
        assert_eq!(remaining_values, expected);

        // Check that we have the expected fragment structure
        let fragments = final_dataset.get_fragments();
        assert_eq!(
            fragments.len(),
            1,
            "Should have one fragment with deletion vector"
        );

        // The fragment should have a deletion vector with 50 deleted rows
        let deletion_vector = fragments[0].get_deletion_vector().await.unwrap().unwrap();
        assert_eq!(deletion_vector.len(), 50, "Should have 50 deleted rows");

        // Check that the deletion vector contains rows 0-49
        let mut deleted_rows: Vec<u32> = deletion_vector.iter().collect();
        deleted_rows.sort();
        let expected_deleted: Vec<u32> = (0..50).collect();
        assert_eq!(deleted_rows, expected_deleted);
    }

    #[tokio::test]
    #[rstest]
    async fn test_delete_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        use crate::{
            dataset::{InsertBuilder, ReadParams, WriteParams, builder::DatasetBuilder},
            session::Session,
            utils::test::ThrottledStoreWrapper,
        };
        use futures::future::try_join_all;
        use lance_io::object_store::ObjectStoreParams;
        use object_store::throttle::ThrottleConfig;
        use tokio::sync::Barrier;

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::UInt32,
            false,
        )]));
        let concurrency = 3;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(
                0..(concurrency * 10),
            ))],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(1),
                wait_get_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                barrier_ref.wait().await;

                // Each task deletes a different range of rows to avoid complete overlap
                let start = i * 10;
                let end = (i + 1) * 10;
                DeleteBuilder::new(
                    Arc::new(dataset),
                    format!("id >= {} AND id < {}", start, end),
                )
                .conflict_retries(5)
                .execute()
                .await
                .unwrap()
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        // All rows should be deleted since each task deleted a non-overlapping range
        let remaining_count = dataset.count_rows(None).await.unwrap();
        assert_eq!(remaining_count, 0, "All rows should be deleted");

        // Verify no fragments remain or they are all empty
        let fragments = dataset.get_fragments();
        if !fragments.is_empty() {
            // If fragments exist, they should all have deletion vectors covering all rows
            for fragment in &fragments {
                let deletion_vector = fragment.get_deletion_vector().await.unwrap();
                assert!(
                    deletion_vector.is_some(),
                    "Fragment should have deletion vector if any rows remain"
                );
            }
        }
    }

    #[tokio::test]
    #[rstest]
    async fn test_delete_true_update_conflict(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::UInt32, false),
            ArrowField::new("value", DataType::UInt32, false),
        ]));

        // Create two batches to ensure multiple fragments
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..100)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(100, 100))),
            ],
        )
        .unwrap();

        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids,
                max_rows_per_file: 50,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();

        // Verify we have 2 fragments initially
        assert_eq!(dataset.get_fragments().len(), 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 100);

        let dataset_arc = Arc::new(dataset);
        let delete_job = DeleteJob {
            dataset: dataset_arc.clone(),
            filter: ExprFilter::Sql("true".to_string()),
            target_fragments: None,
        };
        let delete_data = delete_job.execute_impl().await.unwrap();

        // Verify delete preparation captured all fragments for deletion
        assert_eq!(delete_data.deleted_fragment_ids.len(), 2);
        assert!(delete_data.updated_fragments.is_empty());

        // Run a concurrent update operation that commits
        let update_job = UpdateBuilder::new(dataset_arc.clone())
            .update_where("id < 25")
            .unwrap() // Update first 25 rows
            .set("value", "value + 1000")
            .unwrap()
            .build()
            .unwrap();
        let update_result = update_job.execute().await.unwrap();
        assert_eq!(
            update_result.new_dataset.count_rows(None).await.unwrap(),
            100
        );

        // Now try to commit the delete operation using the stale dataset reference
        // This should fail because the delete was planning to delete fragments that
        // have been modified by the update
        let result = delete_job.commit(dataset_arc.clone(), delete_data).await;

        // When deleting everything with delete("true"), the operation should succeed
        // but it might not delete all rows if concurrent updates moved some rows
        assert!(
            matches!(&result, Err(Error::RetryableCommitConflict { .. })),
            "Expected retryable conflict due to concurrent update, got {:?}",
            result
        );

        // Also verify with the retry mechanism that it works correctly
        let final_result = DeleteBuilder::new(dataset_arc, "true")
            .conflict_retries(5)
            .execute()
            .await
            .unwrap();
        // All rows should be deleted, including the updated ones
        assert_eq!(final_result.new_dataset.count_rows(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_delete_with_expr_filter() {
        use datafusion::prelude::{col, lit};

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100u32))],
        )
        .unwrap();

        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap();

        // Delete rows where i < 10 using an Expr filter
        let expr = col("i").lt(lit(10u32));
        let result = DeleteBuilder::from_expr(Arc::new(dataset.clone()), expr)
            .execute()
            .await
            .unwrap();

        assert_eq!(result.num_deleted_rows, 10);

        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 90);
    }

    // ------------------------------------------------------------------
    // Fragment-scoped / distributed delete
    // ------------------------------------------------------------------

    fn scoped_delete_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]))
    }

    fn scoped_delete_batch(range: Range<u32>) -> RecordBatch {
        RecordBatch::try_new(
            scoped_delete_schema(),
            vec![Arc::new(UInt32Array::from_iter_values(range))],
        )
        .unwrap()
    }

    /// Build a dataset with one fragment per 100-row block: [0,100), [100,200), ...
    async fn make_multi_fragment_dataset(tmp_path: &str, num_fragments: u32) -> Dataset {
        let batches: Vec<RecordBatch> = (0..num_fragments)
            .map(|f| scoped_delete_batch((f * 100)..((f + 1) * 100)))
            .collect();
        TestDatasetGenerator::new(batches, LanceFileVersion::Stable)
            .make_hostile(tmp_path)
            .await
    }

    async fn surviving_ids(dataset: &Dataset) -> Vec<u32> {
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let mut ids: Vec<u32> = batch["i"].as_primitive::<UInt32Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    /// A fragment-scoped delete over disjoint slices, batch-committed, produces
    /// exactly the same surviving rows as a single full delete of the same
    /// predicate.
    #[tokio::test]
    async fn test_fragment_scoped_delete_equals_full_delete() {
        let predicate = "i % 3 = 0";

        // Baseline: full delete over the whole dataset.
        let baseline_dir = TempStrDir::default();
        let mut baseline = make_multi_fragment_dataset(baseline_dir.as_str(), 4).await;
        let baseline_deleted = baseline.delete(predicate).await.unwrap().num_deleted_rows;
        let expected_ids = surviving_ids(&baseline).await;

        // Split: partition the 4 fragments into 2 disjoint slices, delete each
        // slice uncommitted, then batch-commit the two transactions together.
        let split_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(split_dir.as_str(), 4).await);
        let all_ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();
        assert_eq!(all_ids.len(), 4);
        let slices = [vec![all_ids[0], all_ids[1]], vec![all_ids[2], all_ids[3]]];

        let mut transactions = Vec::new();
        for slice in &slices {
            let staged = DeleteBuilder::new(dataset.clone(), predicate)
                .with_target_fragments(slice.clone())
                .execute_uncommitted()
                .await
                .unwrap();

            // Each slice's transaction only touches fragments in that slice.
            let touched = touched_fragment_ids(&staged.transaction);
            for id in &touched {
                assert!(
                    slice.contains(&(*id as u32)),
                    "slice {slice:?} produced out-of-slice fragment {id}"
                );
            }
            transactions.push(staged.transaction);
        }

        // The combiner reconstructs the same total row count and affected-row
        // set as the single full delete.
        let combined = combine_delete_transactions(dataset.as_ref(), transactions.clone())
            .await
            .unwrap();
        assert_eq!(combined.num_deleted_rows, baseline_deleted);
        assert_eq!(
            combined.affected_rows.len(),
            Some(baseline_deleted),
            "affected_rows should cover exactly the deleted rows"
        );

        let result = CommitBuilder::new(dataset.clone())
            .execute_batch(transactions)
            .await
            .unwrap();
        assert!(matches!(result.merged.operation, Operation::Delete { .. }));

        let committed = result.dataset;
        assert_eq!(surviving_ids(&committed).await, expected_ids);
        committed.validate().await.unwrap();
    }

    /// Collect every fragment id referenced by a delete transaction (both
    /// updated and wholly-deleted fragments).
    fn touched_fragment_ids(transaction: &Transaction) -> Vec<u64> {
        match &transaction.operation {
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => updated_fragments
                .iter()
                .map(|f| f.id)
                .chain(deleted_fragment_ids.iter().copied())
                .collect(),
            other => panic!("expected delete transaction, got {other:?}"),
        }
    }

    /// A predicate that matches every row in a slice's fragment removes that
    /// fragment entirely (it lands in `deleted_fragment_ids`, not
    /// `updated_fragments`), and the batch commit still produces the correct
    /// surviving rows.
    #[tokio::test]
    async fn test_fragment_scoped_delete_deletes_whole_fragment() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 3).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        // A single predicate that matches ALL of fragment 0 (rows 0..100) but
        // only a subset of fragments 1 and 2 (rows 150..250). Slice A (fragment
        // 0) therefore deletes its fragment whole; slice B (fragments 1, 2)
        // produces partial updates.
        let predicate = "i < 100 OR (i >= 150 AND i < 250)";

        let staged_a = DeleteBuilder::new(dataset.clone(), predicate)
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        match &staged_a.transaction.operation {
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                ..
            } => {
                assert!(
                    updated_fragments.is_empty(),
                    "whole-fragment delete should not leave an updated fragment"
                );
                assert_eq!(deleted_fragment_ids, &vec![ids[0] as u64]);
            }
            other => panic!("expected delete, got {other:?}"),
        }

        let staged_b = DeleteBuilder::new(dataset.clone(), predicate)
            .with_target_fragments(vec![ids[1], ids[2]])
            .execute_uncommitted()
            .await
            .unwrap();

        let committed = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![staged_a.transaction, staged_b.transaction])
            .await
            .unwrap()
            .dataset;

        // Fragment 0 gone entirely; 150..250 gone; everything else survives.
        let expected: Vec<u32> = (100..150).chain(250..300).collect();
        assert_eq!(surviving_ids(&committed).await, expected);
        committed.validate().await.unwrap();
    }

    /// A slice whose predicate matches nothing contributes an empty delete, and
    /// combining it with a non-empty slice still yields the correct result.
    #[tokio::test]
    async fn test_fragment_scoped_delete_empty_slice() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        // Slice for fragment 0: predicate "i >= 100" matches nothing in [0,100).
        let staged_empty = DeleteBuilder::new(dataset.clone(), "i >= 100")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(staged_empty.num_deleted_rows, 0);
        assert!(touched_fragment_ids(&staged_empty.transaction).is_empty());

        // Slice for fragment 1: delete 150..200.
        let staged_full = DeleteBuilder::new(dataset.clone(), "i >= 100")
            .with_target_fragments(vec![ids[1]])
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(staged_full.num_deleted_rows, 100);

        let committed = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![staged_empty.transaction, staged_full.transaction])
            .await
            .unwrap()
            .dataset;

        let expected: Vec<u32> = (0..100).collect();
        assert_eq!(surviving_ids(&committed).await, expected);
        committed.validate().await.unwrap();
    }

    /// with_target_fragments rejects an id that does not exist in the dataset.
    #[tokio::test]
    async fn test_fragment_scoped_delete_unknown_fragment_errors() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);

        let result = DeleteBuilder::new(dataset, "i < 10")
            .with_target_fragments(vec![999])
            .execute_uncommitted()
            .await;
        assert!(
            matches!(&result, Err(Error::InvalidInput { .. })),
            "expected InvalidInput for unknown fragment id, got {result:?}"
        );
    }

    /// Assert an error is `InvalidInput` and its message contains `needle`, so
    /// the test distinguishes *which* invariant fired (all guards here map to
    /// the same `InvalidInput` variant).
    fn assert_invalid_input_contains(err: &Result<impl std::fmt::Debug>, needle: &str) {
        match err {
            Err(Error::InvalidInput { source, .. }) => {
                let msg = source.to_string();
                assert!(
                    msg.contains(needle),
                    "expected InvalidInput message containing {needle:?}, got {msg:?}"
                );
            }
            other => panic!("expected InvalidInput containing {needle:?}, got {other:?}"),
        }
    }

    /// combine_delete_transactions rejects transactions whose fragment slices
    /// overlap (two tasks both modified the same fragment).
    #[tokio::test]
    async fn test_combine_delete_rejects_overlapping_slices() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        // Both slices target the SAME fragment 0 (overlapping slices), which the
        // combiner must reject before ever touching deletion vectors.
        let staged_a = DeleteBuilder::new(dataset.clone(), "i < 10")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        let staged_b = DeleteBuilder::new(dataset.clone(), "i < 10")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();

        let err = combine_delete_transactions(
            dataset.as_ref(),
            vec![staged_a.transaction, staged_b.transaction],
        )
        .await;
        assert_invalid_input_contains(&err, "not disjoint");
    }

    /// combine_delete_transactions rejects overlapping WHOLE-fragment deletions
    /// (same fragment id in two transactions' deleted_fragment_ids), not just
    /// overlapping partial updates.
    #[tokio::test]
    async fn test_combine_delete_rejects_overlapping_whole_fragment() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        // "i < 100" deletes ALL of fragment 0, so both staged deletes put
        // fragment 0 in deleted_fragment_ids (empty updated_fragments).
        let staged_a = DeleteBuilder::new(dataset.clone(), "i < 100")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        let staged_b = DeleteBuilder::new(dataset.clone(), "i < 100")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        assert!(matches!(
            &staged_a.transaction.operation,
            Operation::Delete { updated_fragments, deleted_fragment_ids, .. }
                if updated_fragments.is_empty() && deleted_fragment_ids == &[ids[0] as u64]
        ));

        let err = combine_delete_transactions(
            dataset.as_ref(),
            vec![staged_a.transaction, staged_b.transaction],
        )
        .await;
        assert_invalid_input_contains(&err, "not disjoint");
    }

    /// combine_delete_transactions rejects a mix of predicates.
    #[tokio::test]
    async fn test_combine_delete_rejects_mismatched_predicate() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        let staged_a = DeleteBuilder::new(dataset.clone(), "i < 10")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        let staged_b = DeleteBuilder::new(dataset.clone(), "i >= 150")
            .with_target_fragments(vec![ids[1]])
            .execute_uncommitted()
            .await
            .unwrap();

        let err = combine_delete_transactions(
            dataset.as_ref(),
            vec![staged_a.transaction, staged_b.transaction],
        )
        .await;
        assert_invalid_input_contains(&err, "same predicate");
    }

    /// with_target_fragments rejects a repeated fragment id.
    #[tokio::test]
    async fn test_fragment_scoped_delete_duplicate_id_errors() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        let err = DeleteBuilder::new(dataset, "i < 10")
            .with_target_fragments(vec![ids[0], ids[0]])
            .execute_uncommitted()
            .await;
        assert_invalid_input_contains(&err, "duplicate fragment id");
    }

    /// execute_batch rejects a batch that mixes operation kinds.
    #[tokio::test]
    async fn test_execute_batch_rejects_mixed_operations() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 2).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        let staged_delete = DeleteBuilder::new(dataset.clone(), "i < 10")
            .with_target_fragments(vec![ids[0]])
            .execute_uncommitted()
            .await
            .unwrap();
        let append = Transaction::new(
            dataset.manifest.version,
            Operation::Append {
                fragments: Vec::new(),
            },
            None,
        );

        let err = CommitBuilder::new(dataset.clone())
            .execute_batch(vec![staged_delete.transaction, append])
            .await
            .err();
        assert!(
            matches!(&err, Some(Error::NotSupported { .. })),
            "expected NotSupported for mixed batch, got {err:?}"
        );
    }

    /// A fragment-scoped distributed delete keeps a scalar index consistent:
    /// the surviving rows match a full delete and post-delete queries are
    /// correct.
    #[tokio::test]
    async fn test_fragment_scoped_delete_with_scalar_index() {
        let predicate = "i % 2 = 0";

        let tmp_dir = TempStrDir::default();
        let mut dataset = make_multi_fragment_dataset(tmp_dir.as_str(), 4).await;
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar_index".to_string()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let dataset = Arc::new(dataset);

        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();
        let slices = [vec![ids[0], ids[1]], vec![ids[2], ids[3]]];
        let mut transactions = Vec::new();
        for slice in &slices {
            let staged = DeleteBuilder::new(dataset.clone(), predicate)
                .with_target_fragments(slice.clone())
                .execute_uncommitted()
                .await
                .unwrap();
            transactions.push(staged.transaction);
        }

        let committed = CommitBuilder::new(dataset.clone())
            .execute_batch(transactions)
            .await
            .unwrap()
            .dataset;
        committed.validate().await.unwrap();

        // Only odd i survive.
        let expected: Vec<u32> = (0..400).filter(|v| v % 2 == 1).collect();
        assert_eq!(surviving_ids(&committed).await, expected);

        // The scalar index survives the distributed delete and still covers the
        // dataset: statistics report the post-delete live-row count with nothing
        // left unindexed. (A filtered count alone would pass even if the index
        // were dropped, since the scanner falls back to a full scan.)
        let stats: serde_json::Value =
            serde_json::from_str(&committed.index_statistics("scalar_index").await.unwrap())
                .unwrap();
        assert_eq!(stats["num_indexed_rows"].as_u64(), Some(200));
        assert_eq!(stats["num_unindexed_rows"].as_u64(), Some(0));

        // An indexed lookup of a deleted value returns nothing; a surviving one
        // returns exactly one row.
        let mut deleted_scan = committed.scan();
        deleted_scan.filter("i = 2").unwrap();
        assert_eq!(deleted_scan.count_rows().await.unwrap(), 0);

        let mut surviving_scan = committed.scan();
        surviving_scan.filter("i = 3").unwrap();
        assert_eq!(surviving_scan.count_rows().await.unwrap(), 1);
    }

    /// A batched distributed delete rebases against a concurrent delete that
    /// touched a different fragment, thanks to the reconstructed affected_rows —
    /// matching a single delete's row-level conflict handling.
    #[tokio::test]
    async fn test_batch_delete_rebases_against_concurrent_delete() {
        let tmp_dir = TempStrDir::default();
        let dataset = Arc::new(make_multi_fragment_dataset(tmp_dir.as_str(), 3).await);
        let ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

        // Stage a distributed delete over fragments 0 and 1 at the current
        // version.
        let mut transactions = Vec::new();
        for id in [ids[0], ids[1]] {
            let staged = DeleteBuilder::new(dataset.clone(), "i % 2 = 0")
                .with_target_fragments(vec![id])
                .execute_uncommitted()
                .await
                .unwrap();
            transactions.push(staged.transaction);
        }

        // A concurrent delete commits first, tombstoning rows in fragment 2 (a
        // fragment the staged batch does NOT touch). This advances the version,
        // so the batch commit must rebase rather than fail.
        DeleteBuilder::new(dataset.clone(), "i >= 250 AND i < 260")
            .execute()
            .await
            .unwrap();

        // The batch commit carries affected_rows, so it rebases onto the new
        // version and succeeds (a None affected_rows would hard-fail here).
        let committed = CommitBuilder::new(dataset.clone())
            .execute_batch(transactions)
            .await
            .unwrap()
            .dataset;
        committed.validate().await.unwrap();

        // Both deletes applied: even i in [0,200) gone (distributed), 250..260
        // gone (concurrent).
        let expected: Vec<u32> = (0..200)
            .filter(|v| v % 2 == 1)
            .chain(200..250)
            .chain(260..300)
            .collect();
        assert_eq!(surviving_ids(&committed).await, expected);
    }
}
