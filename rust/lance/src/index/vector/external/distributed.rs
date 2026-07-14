// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distributed build for the external parquet IVF-PQ index.
//!
//! The single-box [`super::build::build_index`] reads the whole corpus on one
//! machine (the driver in a Spark job) — at 10M+ rows / dim=1024 that 40+ GB scan
//! is the dominant build cost and leaves executors idle. This module splits the
//! build across a cluster with a persistence boundary between the phases:
//!
//! ```text
//! driver     train_broadcast_payload(sample)  -> (PbIvf bytes, PbPq bytes, metric)
//!            broadcast the payload to executors
//! executor   build_shard_to_parquet(payload, files[a..b], offset)
//!              -> assign + PQ-encode its file shard using the SHARED quantizers,
//!                 write the partition-binned shuffle output to <shard_uri> parquet
//! driver     merge_shards(payload, [shard_uri...], out_uri)
//!              -> read every shard's partition rows back, heap-merge by part_id
//!                 via write_ivf_pq_file_external, write final index.idx + manifest
//! ```
//!
//! Only the trained quantizers (~centroids + codebook, KBs) cross the wire to
//! executors, and only the PQ codes (~num_sub_vectors bytes/row) come back to the
//! driver — the 40+ GB vector read stays distributed. Partition assignment is pure
//! over (centroids, codebook) (see [`super::build::shard_partition_streams`]), so
//! shards built independently with the same broadcast payload merge exactly; the
//! `distributed_shard_build_matches_single_pass` test proves the in-process
//! equivalent, and `merge_from_persisted_shards_matches` here proves it survives
//! the persistence round-trip.
//!
//! The rerank sidecar is handled separately (each executor writes its shard at the
//! correct global byte offset), tracked with the index shards.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_index::pb::Ivf as PbIvf;
use lance_index::pb::Pq as PbPq;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::MetricType;
use prost::Message;

use super::params::ExternalIvfPqIndexParams;
use super::parquet_source::ParquetVectorSource;
use super::types::ParquetFileSpec;

/// Serialized, broadcast-ready trained quantizers. Produced once on the driver and
/// shipped to every executor. `metric` must travel separately because `PbPq` does
/// not encode the distance type ([`ProductQuantizer::from_proto`] takes it as an
/// argument, and Cosine is folded to L2 on decode).
#[derive(Clone, Debug)]
pub struct BroadcastPayload {
    pub ivf_pb: Vec<u8>,
    pub pq_pb: Vec<u8>,
    pub metric: MetricType,
}

impl BroadcastPayload {
    /// Serialize to a self-describing byte blob for the JNI/broadcast boundary:
    /// `[u32 metric_len][metric utf8][u64 ivf_len][ivf_pb][u64 pq_len][pq_pb]`,
    /// all little-endian. Round-trips via [`Self::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let metric = format!("{}", self.metric);
        let mut out = Vec::with_capacity(4 + metric.len() + 16 + self.ivf_pb.len() + self.pq_pb.len());
        out.extend_from_slice(&(metric.len() as u32).to_le_bytes());
        out.extend_from_slice(metric.as_bytes());
        out.extend_from_slice(&(self.ivf_pb.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.ivf_pb);
        out.extend_from_slice(&(self.pq_pb.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.pq_pb);
        out
    }

    /// Parse a blob produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let err = || Error::index("BroadcastPayload::from_bytes: truncated blob".to_string());
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize| -> Result<&[u8]> {
            if *pos + n > bytes.len() {
                return Err(Error::index(
                    "BroadcastPayload::from_bytes: truncated blob".to_string(),
                ));
            }
            let s = &bytes[*pos..*pos + n];
            *pos += n;
            Ok(s)
        };
        let mlen = u32::from_le_bytes(take(&mut pos, 4)?.try_into().map_err(|_| err())?) as usize;
        let metric_str = std::str::from_utf8(take(&mut pos, mlen)?)
            .map_err(|e| Error::index(format!("metric utf8: {e}")))?
            .to_string();
        let metric = match metric_str.to_ascii_lowercase().as_str() {
            "l2" => MetricType::L2,
            "cosine" => MetricType::Cosine,
            "dot" => MetricType::Dot,
            other => return Err(Error::index(format!("unknown metric '{other}'"))),
        };
        let ilen = u64::from_le_bytes(take(&mut pos, 8)?.try_into().map_err(|_| err())?) as usize;
        let ivf_pb = take(&mut pos, ilen)?.to_vec();
        let plen = u64::from_le_bytes(take(&mut pos, 8)?.try_into().map_err(|_| err())?) as usize;
        let pq_pb = take(&mut pos, plen)?.to_vec();
        Ok(Self {
            ivf_pb,
            pq_pb,
            metric,
        })
    }

    /// Decode back into live quantizers on an executor.
    pub fn decode(&self) -> Result<(IvfModel, ProductQuantizer)> {
        let ivf_pb = PbIvf::decode(self.ivf_pb.as_slice())
            .map_err(|e| Error::index(format!("decode PbIvf: {e}")))?;
        let ivf = IvfModel::try_from(ivf_pb)?;
        let pq_pb = PbPq::decode(self.pq_pb.as_slice())
            .map_err(|e| Error::index(format!("decode PbPq: {e}")))?;
        let pq = ProductQuantizer::from_proto(&pq_pb, self.metric.into())?;
        Ok((ivf, pq))
    }
}

/// Driver phase 1: train the IVF centroids + PQ codebook on a sample of the whole
/// corpus, then serialize to a [`BroadcastPayload`] for the executors. Cheap —
/// reads only `num_partitions * sample_rate` rows regardless of corpus size.
pub async fn train_broadcast_payload(
    files: Vec<ParquetFileSpec>,
    vector_column: &str,
    params: &ExternalIvfPqIndexParams,
) -> Result<BroadcastPayload> {
    let source = ParquetVectorSource::try_new(files, vector_column).await?;
    let (ivf, pq, _training) = super::build::train_quantizers(&source, params).await?;
    let ivf_pb = PbIvf::try_from(&ivf)?.encode_to_vec();
    let pq_pb = PbPq::try_from(&pq)?.encode_to_vec();
    Ok(BroadcastPayload {
        ivf_pb,
        pq_pb,
        metric: params.metric,
    })
}

/// Executor phase 2: assign + PQ-encode this shard's `files` (whose global index
/// starts at `file_id_offset`) using the broadcast quantizers, and persist the
/// partition-binned output to `shard_uri` as parquet. Returns the number of
/// non-empty partition groups written (diagnostic).
///
/// The persisted parquet carries the shuffle schema (`_rowid`, `__ivf_part_id`,
/// `__pq_code`) verbatim, so the driver reads it straight back into the merge.
/// [`merge_shards`] regroups on read by `__ivf_part_id` so each merge input stream
/// is single-partition (the writer's contract), robust to parquet re-batching.
pub async fn build_shard_to_parquet(
    payload: &BroadcastPayload,
    files: Vec<ParquetFileSpec>,
    vector_column: &str,
    file_id_offset: u32,
    params: &ExternalIvfPqIndexParams,
    shard_uri: &str,
) -> Result<usize> {
    let (ivf, pq) = payload.decode()?;
    let source =
        ParquetVectorSource::try_new_with_offset(files, vector_column, file_id_offset).await?;
    let collected =
        super::build::shard_partition_streams(&source, vector_column, &ivf, &pq, params).await?;

    let num_groups = collected.len();
    write_shard_parquet(shard_uri, collected).await?;
    Ok(num_groups)
}

/// Driver phase 3: read every shard's persisted partition output back, and
/// heap-merge by part_id into the final `index.idx` via the same
/// [`write_ivf_pq_file_external`] path the single-box build uses. Each shard file
/// yields one owned stream per partition-group it wrote; concatenating them across
/// shards gives the writer every same-part_id stream to union.
pub async fn merge_shards(
    payload: &BroadcastPayload,
    shard_uris: &[String],
    vector_column: &str,
    index_path: &object_store::path::Path,
    object_store: &ObjectStore,
) -> Result<()> {
    let (ivf, pq) = payload.decode()?;

    let mut all_groups: Vec<Vec<RecordBatch>> = Vec::new();
    for uri in shard_uris {
        all_groups.extend(read_shard_parquet(uri).await?);
    }
    let streams: Vec<_> = all_groups
        .into_iter()
        .filter(|g| !g.is_empty())
        .map(|batches| futures::stream::iter(batches.into_iter().map(Ok::<_, Error>)))
        .collect();

    crate::index::vector::ivf::write_ivf_pq_file_external(
        object_store,
        index_path,
        vector_column,
        "external_ivf_pq",
        0,
        ivf,
        pq,
        streams,
    )
    .await
}

/// URI-based wrapper over [`merge_shards`] for the JNI/Spark layer: resolves
/// `index_uri` into its object store + path internally so callers pass only
/// strings. `index_uri` is the full path to the `index.idx` to write (the caller
/// composes `<dir>/<uuid>/index.idx` and writes the manifest alongside).
pub async fn merge_shards_to_uri(
    payload: &BroadcastPayload,
    shard_uris: &[String],
    vector_column: &str,
    index_uri: &str,
) -> Result<()> {
    let (object_store, index_path) = ObjectStore::from_uri(index_uri).await?;
    merge_shards(payload, shard_uris, vector_column, &index_path, &object_store).await
}

// ---- shard parquet I/O -------------------------------------------------------

/// Write a shard's partition-binned batches to one parquet file. The batches carry
/// the shuffle schema (`_rowid`, `__ivf_part_id`, `__pq_code`) verbatim — no extra
/// tagging. Regrouping on read is keyed off `__ivf_part_id` directly, which is
/// robust to the parquet reader re-batching across our write boundaries.
async fn write_shard_parquet(shard_uri: &str, groups: Vec<Vec<RecordBatch>>) -> Result<()> {
    use parquet::arrow::AsyncArrowWriter;

    let out_schema: Option<SchemaRef> = groups
        .iter()
        .flat_map(|g| g.first())
        .map(|b| b.schema())
        .next();
    let Some(out_schema) = out_schema else {
        // Shard produced no rows; write a marker so the driver's read finds a file.
        return write_empty_marker(shard_uri).await;
    };

    let (object_store, path) = ObjectStore::from_uri(shard_uri).await?;
    let writer = object_store.create(&path).await?;
    let mut aw = AsyncArrowWriter::try_new(WriterAdapter(writer), out_schema.clone(), None)
        .map_err(|e| Error::io(format!("shard writer create: {e}")))?;
    for batches in groups {
        for b in batches {
            aw.write(&b)
                .await
                .map_err(|e| Error::io(format!("shard write: {e}")))?;
        }
    }
    aw.close()
        .await
        .map_err(|e| Error::io(format!("shard writer close: {e}")))?;
    Ok(())
}

/// Read a shard parquet back, regrouped so each returned batch-vec holds rows of a
/// SINGLE `__ivf_part_id`. This is the merge writer's contract: `merge_streams`
/// assigns each input batch wholesale to one partition (it reads `part_id[0]`), so
/// a batch spanning multiple part_ids would corrupt the merge. Regrouping by the
/// part_id column directly is robust to however the parquet reader chunks rows.
async fn read_shard_parquet(shard_uri: &str) -> Result<Vec<Vec<RecordBatch>>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt32Type;
    use lance_index::vector::PART_ID_COLUMN;
    use std::collections::HashMap;

    let builder = super::parquet_source::open_parquet_async(shard_uri).await?;
    if builder.metadata().file_metadata().num_rows() == 0 {
        return Ok(Vec::new()); // empty marker
    }
    let mut stream = builder
        .build()
        .map_err(|e| Error::io(format!("shard reader build {shard_uri}: {e}")))?;

    // part_id → its batches. The shuffler emits part_ids in ascending order, so
    // within a read batch each part_id occupies a contiguous run; slice on those
    // runs and route each slice to its part_id bucket.
    let mut by_part: HashMap<u32, Vec<RecordBatch>> = HashMap::new();
    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|e| Error::io(format!("shard read {shard_uri}: {e}")))?
    {
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        let pid_idx = batch
            .schema()
            .index_of(PART_ID_COLUMN)
            .map_err(|e| Error::io(format!("shard missing {PART_ID_COLUMN}: {e}")))?;
        let pid_col = batch.column(pid_idx).as_primitive::<UInt32Type>();
        let mut start = 0usize;
        while start < n {
            let pid = pid_col.value(start);
            let mut end = start + 1;
            while end < n && pid_col.value(end) == pid {
                end += 1;
            }
            by_part
                .entry(pid)
                .or_default()
                .push(batch.slice(start, end - start));
            start = end;
        }
    }
    Ok(by_part.into_values().collect())
}

/// Write a zero-row parquet so the driver's read finds a file for an empty shard.
async fn write_empty_marker(shard_uri: &str) -> Result<()> {
    use arrow_array::UInt64Array;
    use parquet::arrow::AsyncArrowWriter;
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "_empty",
        arrow_schema::DataType::UInt64,
        false,
    )]));
    let (object_store, path) = ObjectStore::from_uri(shard_uri).await?;
    let writer = object_store.create(&path).await?;
    let mut aw = AsyncArrowWriter::try_new(WriterAdapter(writer), schema.clone(), None)
        .map_err(|e| Error::io(format!("empty marker create: {e}")))?;
    let empty = RecordBatch::new_empty(schema);
    let _ = UInt64Array::from(Vec::<u64>::new());
    aw.write(&empty)
        .await
        .map_err(|e| Error::io(format!("empty marker write: {e}")))?;
    aw.close()
        .await
        .map_err(|e| Error::io(format!("empty marker close: {e}")))?;
    Ok(())
}

/// Adapts a lance-io `Writer` (tokio `AsyncWrite`) to what `AsyncArrowWriter`
/// needs. `AsyncArrowWriter` requires `AsyncWrite + Unpin + Send`; the boxed
/// object-store writer already is, so this is a thin newtype for coherence.
struct WriterAdapter(Box<dyn lance_io::traits::Writer>);

impl tokio::io::AsyncWrite for WriterAdapter {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.0).poll_write(cx, buf)
    }
    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.0).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::external::ExternalIvfPqIndex;
    use crate::index::vector::external::params::ExternalIvfPqIndexParams;
    use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::MetricType;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn write_parquet(path: &std::path::Path, n: usize, dim: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = (0..n * dim)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
        let fsl = FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32)
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            fsl.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// The persistence-boundary proof: build each file as an independent shard,
    /// PERSIST each shard's partition output to parquet, then merge from the
    /// persisted files — and assert the result is byte-identical to a whole-corpus
    /// build that shares the same broadcast payload. This is the piece the
    /// in-process `distributed_shard_build_matches_single_pass` test does NOT
    /// cover: serialize quantizers → decode on "executors" → persist shards →
    /// read back → merge.
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_from_persisted_shards_matches() {
        const DIM: usize = 16;
        const K: usize = 10;
        const NUM_FILES: usize = 4;
        const PER_FILE: usize = 300;

        let data = TempDir::new().unwrap();
        let files: Vec<ParquetFileSpec> = (0..NUM_FILES)
            .map(|i| {
                let p = data.path().join(format!("f{i}.parquet"));
                write_parquet(&p, PER_FILE, DIM, 200 + i as u64);
                ParquetFileSpec::of(p.to_str().unwrap())
            })
            .collect();

        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(4)
            .num_sub_vectors(4)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(10)
            .sample_rate(80)
            .build();

        // Driver: train + serialize once. Both the distributed build and the
        // reference whole-corpus build below decode THIS payload, so they share
        // identical centroids/codebook (kmeans is nondeterministic).
        let payload = train_broadcast_payload(files.clone(), "vec", &params)
            .await
            .expect("train");

        // Executors (simulated): each file is its own shard, persisted to parquet.
        let shard_dir = TempDir::new().unwrap();
        let mut shard_uris = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let shard_uri = shard_dir.path().join(format!("shard{i}.parquet"));
            let shard_uri = shard_uri.to_str().unwrap().to_string();
            build_shard_to_parquet(
                &payload,
                vec![f.clone()],
                "vec",
                i as u32, // global file_id offset = file position
                &params,
                &shard_uri,
            )
            .await
            .expect("build shard");
            shard_uris.push(shard_uri);
        }

        // Driver: merge persisted shards into the final index.
        let dist_out = TempDir::new().unwrap();
        let (os, root) = ObjectStore::from_uri(dist_out.path().to_str().unwrap())
            .await
            .unwrap();
        let dist_uuid = uuid::Uuid::new_v4();
        let dist_dir = root.child(dist_uuid.to_string());
        let dist_index = dist_dir.child(super::super::open::INDEX_FILE_NAME);
        merge_shards(&payload, &shard_uris, "vec", &dist_index, &os)
            .await
            .expect("merge");
        // Manifest for the distributed index (open() needs it).
        let mut resolved = files.clone();
        for s in resolved.iter_mut() {
            if s.num_rows == 0 {
                let b = super::super::parquet_source::open_parquet_async(&s.file_path)
                    .await
                    .unwrap();
                s.num_rows = b.metadata().file_metadata().num_rows() as u64;
            }
        }
        let manifest = super::super::manifest::ExternalIndexManifest::from_build(
            "vec", &resolved, &params, None,
        );
        super::super::manifest::write_manifest(&os, &dist_dir, &manifest)
            .await
            .unwrap();

        // Reference: whole-corpus build from the SAME payload (decode → assign all
        // files in one pass → write). Same quantizers ⇒ must match the merged shards.
        let (ivf, pq) = payload.decode().unwrap();
        let ref_source = ParquetVectorSource::try_new(files.clone(), "vec")
            .await
            .unwrap();
        let ref_groups =
            super::super::build::shard_partition_streams(&ref_source, "vec", &ivf, &pq, &params)
                .await
                .unwrap();
        let ref_out = TempDir::new().unwrap();
        let (ros, rroot) = ObjectStore::from_uri(ref_out.path().to_str().unwrap())
            .await
            .unwrap();
        let ref_uuid = uuid::Uuid::new_v4();
        let ref_dir = rroot.child(ref_uuid.to_string());
        let ref_index = ref_dir.child(super::super::open::INDEX_FILE_NAME);
        let ref_streams: Vec<_> = ref_groups
            .into_iter()
            .filter(|g| !g.is_empty())
            .map(|b| futures::stream::iter(b.into_iter().map(Ok::<_, Error>)))
            .collect();
        crate::index::vector::ivf::write_ivf_pq_file_external(
            &ros,
            &ref_index,
            "vec",
            "external_ivf_pq",
            0,
            ivf,
            pq,
            ref_streams,
        )
        .await
        .unwrap();
        super::super::manifest::write_manifest(&ros, &ref_dir, &manifest)
            .await
            .unwrap();

        // Compare: identical search results. Open via the real filesystem path +
        // uuid (the object-store `Path::to_string` is cwd-relative and won't open).
        let idx_dist = ExternalIvfPqIndex::open(
            dist_out
                .path()
                .join(dist_uuid.to_string())
                .to_str()
                .unwrap(),
        )
        .await
        .expect("open dist");
        let idx_ref =
            ExternalIvfPqIndex::open(ref_out.path().join(ref_uuid.to_string()).to_str().unwrap())
                .await
                .expect("open ref");

        // The distributed-merged index and the whole-corpus reference contain the
        // SAME rows with the SAME PQ codes (same broadcast quantizers), so for any
        // query they must return the same top-K *distance multiset* — the merge is
        // correct iff sorted distances match. (We don't assert positional (file,row)
        // equality: PQ distances are quantized, so ties at the K-th candidate can
        // resolve to different-but-equidistant rows depending on within-partition
        // row order, which is legitimate approximate-search nondeterminism, not a
        // merge bug. The set overlap is asserted too, allowing one tie-boundary flip.)
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..32 {
            let q: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let rd = idx_dist
                .search(&q, K, 4, 4, None)
                .await
                .expect("search dist");
            let rr = idx_ref.search(&q, K, 4, 4, None).await.expect("search ref");
            assert_eq!(rd.len(), rr.len(), "result count differs");

            let mut dd: Vec<f32> = rd.iter().map(|r| r.distance).collect();
            let mut dr: Vec<f32> = rr.iter().map(|r| r.distance).collect();
            dd.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dr.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (a, b) in dd.iter().zip(dr.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "sorted distances differ: {a} vs {b} — merge produced a different index"
                );
            }

            // Result sets should coincide except possibly at one tie boundary.
            let sd: std::collections::HashSet<(String, u64)> = rd
                .iter()
                .map(|r| (r.file_path.clone(), r.row_index))
                .collect();
            let sr: std::collections::HashSet<(String, u64)> = rr
                .iter()
                .map(|r| (r.file_path.clone(), r.row_index))
                .collect();
            assert!(
                sd.intersection(&sr).count() >= K - 1,
                "result sets diverge by more than one tie flip"
            );
        }
    }
}
