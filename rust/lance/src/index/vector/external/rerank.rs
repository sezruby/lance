// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Co-located SQ8 rerank store for the external parquet index.
//!
//! # Why this exists
//!
//! The IVF-PQ index persists only PQ codes (≈16 B/row). To turn PQ-approximate
//! candidates into a correctly-ordered top-K, refinement needs each candidate's
//! *original* vector. Reading those back from the source parquet page-decodes a
//! multi-MB data page per scattered candidate — the dominant per-query cost on
//! wide (dim≈1024) embeddings.
//!
//! The rerank store trades a bounded amount of build-time storage for a
//! page-decode-free refine read: it writes a scalar-quantized (int8) copy of
//! every indexed vector, row-major by *global ordinal*, so any candidate maps to
//! a contiguous `dim`-byte range. Refinement reads those ranges (coalesced) and
//! reranks with integer [`l2_u8`], never touching the parquet vector column.
//!
//! This is deliberately orthogonal to the coarse index type — "store the vector
//! for reranking" independent of how candidates are found.
//!
//! # Layout (`rerank.sq8`)
//!
//! ```text
//! byte 0                    dim                 2*dim               total_rows*dim
//! ├──── row 0 (dim × u8) ────┼──── row 1 ────────┼── … ──┼──── row N-1 ────┤
//! ```
//!
//! Row `ordinal = global_base(file_id) + row_in_file` (see
//! [`super::manifest::ExternalIndexManifest::global_base`]). No per-row framing;
//! `dim` and the SQ bounds live in [`super::manifest::RerankStoreMeta`].

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{Array, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::DataType;
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_index::vector::sq::ScalarQuantizer;
use lance_io::object_store::ObjectStore;
use lance_io::traits::{Reader, Writer};
use lance_linalg::distance::l2_distance;
use lance_linalg::distance::l2_u8::l2_u8;
use object_store::path::Path;
use tokio::io::AsyncWriteExt;

use super::manifest::{
    ExternalIndexManifest, RERANK_FLAT_FILE_NAME, RERANK_SQ8_FILE_NAME, RerankStoreMeta,
};
use super::parquet_source::{coerce_to_fsl, open_parquet_async};

/// SQ8 uses 8 bits per dimension by definition.
const SQ_NUM_BITS: u16 = 8;

/// Which rerank store a build/open should produce or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RerankKind {
    /// int8 scalar-quantized, `dim` bytes/row, integer `l2_u8` rerank.
    Sq8,
    /// full-precision f32, `dim*4` bytes/row, exact f32 L2 rerank.
    Flat,
}

impl RerankKind {
    fn tag(self) -> &'static str {
        match self {
            RerankKind::Sq8 => "sq8",
            RerankKind::Flat => "flat",
        }
    }
    fn file_name(self) -> &'static str {
        match self {
            RerankKind::Sq8 => RERANK_SQ8_FILE_NAME,
            RerankKind::Flat => RERANK_FLAT_FILE_NAME,
        }
    }
    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "sq8" => Some(RerankKind::Sq8),
            "flat" => Some(RerankKind::Flat),
            _ => None,
        }
    }
}

/// Build a co-located rerank sidecar under `index_dir`.
///
/// Scans every registered parquet file *in manifest order* (which is what makes
/// the write row-major by global ordinal) and appends each row's encoded vector:
/// - [`RerankKind::Sq8`]: `dim` quantized `u8` bytes, using `bounds` (the SQ min/max
///   learned from the build sample); the query is quantized with the identical
///   bounds at search time.
/// - [`RerankKind::Flat`]: `dim * 4` raw little-endian f32 bytes; no quantization,
///   `bounds` ignored.
///
/// Returns the [`RerankStoreMeta`] to record in the manifest.
pub(super) async fn build_store(
    object_store: &ObjectStore,
    index_dir: &Path,
    files: &[super::types::ParquetFileSpec],
    vector_column: &str,
    dim: usize,
    kind: RerankKind,
    bounds: std::ops::Range<f64>,
) -> Result<RerankStoreMeta> {
    let quantizer = match kind {
        RerankKind::Sq8 => Some(ScalarQuantizer::with_bounds(
            SQ_NUM_BITS,
            dim,
            bounds.clone(),
        )),
        RerankKind::Flat => None,
    };

    let path = index_dir.clone().join(kind.file_name());
    let mut writer = object_store.create(&path).await?;

    let mut total_rows: u64 = 0;
    for spec in files {
        // Stream the file's vector column in row order, encode each batch, and append
        // its bytes. Reading one file at a time keeps peak memory at one file's
        // vectors rather than the whole corpus.
        let builder = open_parquet_async(&spec.file_path).await?;
        let projection = parquet_column_projection(&builder, vector_column)?;
        let mut stream = builder
            .with_projection(projection)
            .with_batch_size(8192)
            .build()
            .map_err(|e| {
                Error::invalid_input(format!(
                    "rerank build: parquet reader for {}: {e}",
                    spec.file_path
                ))
            })?;

        while let Some(batch) = stream.try_next().await.map_err(|e| {
            Error::invalid_input(format!("rerank build: read {}: {e}", spec.file_path))
        })? {
            let col = batch.column_by_name(vector_column).ok_or_else(|| {
                Error::invalid_input(format!(
                    "rerank build: column '{vector_column}' missing in {}",
                    spec.file_path
                ))
            })?;
            let fsl = coerce_to_fsl(col, dim)?;
            match &quantizer {
                Some(q) => {
                    let codes = q.transform::<arrow::datatypes::Float32Type>(&fsl)?;
                    let code_bytes = fixed_size_list_u8_values(&codes)?;
                    writer
                        .write_all(code_bytes)
                        .await
                        .map_err(|e| Error::io(format!("rerank build: write {}: {e}", path)))?;
                }
                None => {
                    // Flat: write the raw f32 values as little-endian bytes.
                    let raw = fixed_size_list_f32_values(&fsl)?;
                    let mut buf = Vec::with_capacity(raw.len() * 4);
                    for v in raw {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    writer
                        .write_all(&buf)
                        .await
                        .map_err(|e| Error::io(format!("rerank build: write {}: {e}", path)))?;
                }
            }
            total_rows += fsl.len() as u64;
        }
    }
    Writer::shutdown(writer.as_mut()).await?;

    let (bounds_min, bounds_max) = match kind {
        RerankKind::Sq8 => (Some(bounds.start), Some(bounds.end)),
        RerankKind::Flat => (None, None),
    };
    Ok(RerankStoreMeta {
        kind: kind.tag().to_string(),
        dim,
        total_rows,
        bounds_min,
        bounds_max,
        shards: Vec::new(), // single-file layout
    })
}

/// Distributed sidecar: encode this shard's `files` and write them to `shard_uri`
/// as one contiguous run of `bytes_per_row`-sized rows, in the files' given order.
/// Returns the number of rows written (the shard's `ordinal_count`). The caller
/// records `(shard_uri, ordinal_base, count)` in the manifest; `ordinal_base` is
/// the global ordinal of this shard's first file, so the shards tile the global
/// ordinal space exactly (same layout the single-file store has, just split).
///
/// Runs on an executor beside the index shard build — the same 40+ GB scan feeds
/// both, so distributing the index build and leaving the sidecar on the driver
/// would re-read the corpus; this keeps the whole scan distributed.
pub(super) async fn build_store_shard(
    object_store: &ObjectStore,
    shard_uri: &Path,
    files: &[super::types::ParquetFileSpec],
    vector_column: &str,
    dim: usize,
    kind: RerankKind,
    bounds: std::ops::Range<f64>,
) -> Result<u64> {
    let quantizer = match kind {
        RerankKind::Sq8 => Some(ScalarQuantizer::with_bounds(SQ_NUM_BITS, dim, bounds)),
        RerankKind::Flat => None,
    };
    let mut writer = object_store.create(shard_uri).await?;
    let mut rows: u64 = 0;
    for spec in files {
        let builder = open_parquet_async(&spec.file_path).await?;
        let projection = parquet_column_projection(&builder, vector_column)?;
        let mut stream = builder
            .with_projection(projection)
            .with_batch_size(8192)
            .build()
            .map_err(|e| {
                Error::invalid_input(format!("rerank shard: reader {}: {e}", spec.file_path))
            })?;
        while let Some(batch) = stream.try_next().await.map_err(|e| {
            Error::invalid_input(format!("rerank shard: read {}: {e}", spec.file_path))
        })? {
            let col = batch.column_by_name(vector_column).ok_or_else(|| {
                Error::invalid_input(format!("rerank shard: column '{vector_column}' missing"))
            })?;
            let fsl = coerce_to_fsl(col, dim)?;
            match &quantizer {
                Some(q) => {
                    let codes = q.transform::<arrow::datatypes::Float32Type>(&fsl)?;
                    writer
                        .write_all(fixed_size_list_u8_values(&codes)?)
                        .await
                        .map_err(|e| Error::io(format!("rerank shard write: {e}")))?;
                }
                None => {
                    let raw = fixed_size_list_f32_values(&fsl)?;
                    let mut buf = Vec::with_capacity(raw.len() * 4);
                    for v in raw {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    writer
                        .write_all(&buf)
                        .await
                        .map_err(|e| Error::io(format!("rerank shard write: {e}")))?;
                }
            }
            rows += fsl.len() as u64;
        }
    }
    Writer::shutdown(writer.as_mut()).await?;
    Ok(rows)
}

/// The per-query query representation the reader compares stored rows against:
/// quantized `u8` codes for sq8, or the raw f32 query for flat.
pub(super) enum QueryRepr {
    Sq8(Vec<u8>),
    Flat(Vec<f32>),
}

/// One opened sidecar segment: its object reader plus the global ordinal range it
/// covers. A single-file sidecar is one segment `[0, total_rows)`; a distributed
/// sidecar is one segment per shard. A global ordinal `o` lives in the segment
/// whose `[base, base+count)` contains it, at byte offset `(o - base) * bpr`.
struct Segment {
    reader: Arc<dyn Reader>,
    base: u64,
    count: u64,
}

/// Search-side reader over an opened rerank sidecar (sq8 or flat), single-file or
/// sharded. Holds the segment readers + decoded [`RerankStoreMeta`] so per-query
/// refinement maps global ordinals → (segment, byte range) without re-parsing the
/// manifest.
pub(super) struct RerankReader {
    /// Segments sorted by `base`, tiling `[0, total_rows)` contiguously.
    segments: Vec<Segment>,
    meta: RerankStoreMeta,
    kind: RerankKind,
    /// Quantizer rebuilt from the stored bounds (sq8 only). `None` for flat.
    quantizer: Option<ScalarQuantizer>,
}

impl RerankReader {
    /// Open the sidecar for `meta.kind` under `index_dir`. Opens the single sidecar
    /// file, or every shard file when the meta is sharded. Cheap — no store bytes
    /// are read beyond what `ObjectStore::open` needs.
    pub(super) async fn open(
        object_store: &ObjectStore,
        index_dir: &Path,
        meta: RerankStoreMeta,
    ) -> Result<Self> {
        let kind = RerankKind::from_tag(&meta.kind)
            .ok_or_else(|| Error::index(format!("unknown rerank store kind '{}'", meta.kind)))?;

        let mut segments: Vec<Segment> = if meta.is_sharded() {
            // Distributed: open each shard by its recorded URI.
            let mut segs = Vec::with_capacity(meta.shards.len());
            for shard in &meta.shards {
                let (shard_store, shard_path) = ObjectStore::from_uri(&shard.path).await?;
                let reader: Arc<dyn Reader> = Arc::from(shard_store.open(&shard_path).await?);
                segs.push(Segment {
                    reader,
                    base: shard.ordinal_base,
                    count: shard.ordinal_count,
                });
            }
            segs
        } else {
            // Single-file: one segment covering everything, at index_dir/<file>.
            let path = index_dir.clone().join(kind.file_name());
            let reader: Arc<dyn Reader> = Arc::from(object_store.open(&path).await?);
            vec![Segment {
                reader,
                base: 0,
                count: meta.total_rows,
            }]
        };
        segments.sort_by_key(|s| s.base);

        let quantizer = match kind {
            RerankKind::Sq8 => {
                let lo = meta.bounds_min.ok_or_else(|| {
                    Error::index("sq8 rerank store missing bounds_min".to_string())
                })?;
                let hi = meta.bounds_max.ok_or_else(|| {
                    Error::index("sq8 rerank store missing bounds_max".to_string())
                })?;
                Some(ScalarQuantizer::with_bounds(SQ_NUM_BITS, meta.dim, lo..hi))
            }
            RerankKind::Flat => None,
        };
        Ok(Self {
            segments,
            meta,
            kind,
            quantizer,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.meta.dim
    }

    /// Fetch the encoded rows for `ordinals` (global ordinals). Returns a map from
    /// ordinal → owned `bytes_per_row`-byte row, reading each distinct ordinal once.
    ///
    /// Ordinals are grouped by the segment that owns them, then read per contiguous
    /// run within a segment (one `get_range` per run) — so clustered candidates cost
    /// few round trips, and a sharded sidecar reads each shard independently.
    pub(super) async fn fetch_codes(&self, ordinals: &[u64]) -> Result<HashMap<u64, Vec<u8>>> {
        let bpr = self.meta.bytes_per_row() as u64;
        let bpr_usize = self.meta.bytes_per_row();
        let mut sorted: Vec<u64> = ordinals.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if let Some(&last) = sorted.last() {
            if last >= self.meta.total_rows {
                return Err(Error::invalid_input(format!(
                    "rerank fetch: ordinal {last} out of range ({} rows)",
                    self.meta.total_rows
                )));
            }
        }

        // Build the list of (segment_index, global_run_start, global_run_end) ranges to read.
        // IVF-PQ candidates are scattered, so most "runs" are a single ordinal. We (1) merge
        // ordinals separated by a small gap into one range (reading a few unused rows is far
        // cheaper than an extra object-store round trip), then (2) issue all range reads
        // CONCURRENTLY. Previously these were awaited serially — at ~10 ms/RTT over abfss, 80
        // scattered candidates cost ~0.8 s of pure latency per query; concurrency + gap-merge
        // collapse that to a handful of parallel round trips.
        const MAX_GAP_ROWS: u64 = 64; // merge runs within 64 rows; bounds wasted bytes per read.
        struct Run {
            seg_idx: usize,
            start: u64, // global ordinal (inclusive)
            end: u64,   // global ordinal (exclusive)
        }
        let mut runs: Vec<Run> = Vec::new();
        let mut i = 0usize;
        while i < sorted.len() {
            let run_start = sorted[i];
            let seg_idx = self
                .segments
                .iter()
                .position(|s| run_start >= s.base && run_start < s.base + s.count)
                .ok_or_else(|| {
                    Error::index(format!("rerank fetch: ordinal {run_start} in no shard"))
                })?;
            let seg_end = self.segments[seg_idx].base + self.segments[seg_idx].count;
            // Extend the run while the next ordinal is within MAX_GAP_ROWS and same segment.
            let mut j = i + 1;
            while j < sorted.len()
                && sorted[j] < seg_end
                && sorted[j] - sorted[j - 1] <= MAX_GAP_ROWS
            {
                j += 1;
            }
            runs.push(Run {
                seg_idx,
                start: run_start,
                end: sorted[j - 1] + 1,
            });
            i = j;
        }

        // Issue every run's get_range concurrently, then stitch results.
        let fetches = runs.iter().map(|run| {
            let seg = &self.segments[run.seg_idx];
            let byte_start = ((run.start - seg.base) * bpr) as usize;
            let byte_end = ((run.end - seg.base) * bpr) as usize;
            let reader = seg.reader.clone();
            async move {
                let bytes = reader.get_range(byte_start..byte_end).await?;
                Ok::<(u64, u64, bytes::Bytes), Error>((run.start, run.end, bytes))
            }
        });
        let results = futures::future::try_join_all(fetches).await?;

        let mut out: HashMap<u64, Vec<u8>> = HashMap::with_capacity(sorted.len());
        for (run_start, run_end, bytes) in results {
            for (k, ord) in (run_start..run_end).enumerate() {
                let off = k * bpr_usize;
                out.insert(ord, bytes[off..off + bpr_usize].to_vec());
            }
        }
        Ok(out)
    }

    /// Encode a query into the representation used to compare against fetched rows:
    /// quantized `u8` codes (sq8) via the same [`ScalarQuantizer::transform`] the
    /// build used, or the raw f32 query (flat).
    pub(super) fn encode_query(&self, query: &[f32]) -> Result<QueryRepr> {
        match &self.quantizer {
            Some(q) => {
                let flat = Float32Array::from(query.to_vec());
                let fsl = FixedSizeListArray::try_new_from_values(flat, self.meta.dim as i32)
                    .map_err(|e| Error::index(format!("rerank: query → FSL: {e}")))?;
                let codes = q.transform::<arrow::datatypes::Float32Type>(&fsl)?;
                Ok(QueryRepr::Sq8(fixed_size_list_u8_values(&codes)?.to_vec()))
            }
            None => Ok(QueryRepr::Flat(query.to_vec())),
        }
    }

    /// Distance between the encoded query and a fetched stored row.
    /// - sq8: integer `l2_u8` on the u8 codes (monotone in true L2; ordering only).
    /// - flat: exact f32 L2 on the decoded row.
    ///
    /// Returned as `f32`; callers use it only to rank candidates.
    pub(super) fn distance(&self, query: &QueryRepr, row: &[u8]) -> f32 {
        match (self.kind, query) {
            (RerankKind::Sq8, QueryRepr::Sq8(qc)) => l2_u8(qc, row) as f32,
            (RerankKind::Flat, QueryRepr::Flat(q)) => {
                let row_f32 = bytes_to_f32(row);
                l2_distance(q, &row_f32)
            }
            // Kind/query mismatch is a programmer error — the reader builds both.
            _ => f32::INFINITY,
        }
    }
}

/// Decode a little-endian f32 byte row into a Vec<f32>.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build a `ProjectionMask` selecting only `column` from the parquet schema.
fn parquet_column_projection(
    builder: &parquet::arrow::arrow_reader::ArrowReaderBuilder<impl Sized>,
    column: &str,
) -> Result<parquet::arrow::ProjectionMask> {
    let schema = builder.parquet_schema();
    let idx = builder
        .schema()
        .index_of(column)
        .map_err(|e| Error::invalid_input(format!("rerank build: column '{column}': {e}")))?;
    Ok(parquet::arrow::ProjectionMask::roots(schema, [idx]))
}

/// Extract the raw `u8` value slice from a `FixedSizeList<u8>` produced by the
/// scalar quantizer. Assumes contiguous, offset-0, null-free values (true for
/// freshly-quantized batches).
fn fixed_size_list_u8_values(codes: &dyn Array) -> Result<&[u8]> {
    let fsl = codes
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::index("rerank build: SQ output is not FixedSizeList"))?;
    if !matches!(fsl.value_type(), DataType::UInt8) {
        return Err(Error::index(format!(
            "rerank build: SQ output inner type {:?}, expected UInt8",
            fsl.value_type()
        )));
    }
    let values: &UInt8Array = fsl
        .values()
        .as_primitive_opt()
        .ok_or_else(|| Error::index("rerank build: SQ code values are not UInt8Array"))?;
    Ok(values.values())
}

/// Extract the raw `f32` value slice from a `FixedSizeList<f32>`. Assumes contiguous,
/// offset-0, null-free values (true for a freshly-coerced vector batch).
fn fixed_size_list_f32_values(fsl: &FixedSizeListArray) -> Result<&[f32]> {
    let values: &Float32Array = fsl
        .values()
        .as_primitive_opt()
        .ok_or_else(|| Error::index("rerank build: flat vector values are not Float32Array"))?;
    Ok(values.values())
}

/// Convenience: `Some(meta)` iff the manifest carries a recognized rerank store
/// (`sq8` or `flat`).
pub(super) fn rerank_meta(manifest: &ExternalIndexManifest) -> Option<&RerankStoreMeta> {
    manifest
        .rerank
        .as_ref()
        .filter(|m| RerankKind::from_tag(&m.kind).is_some())
}
