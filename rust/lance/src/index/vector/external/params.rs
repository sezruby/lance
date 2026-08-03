// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build parameters for [`super::ExternalIvfPqIndex`].

use lance_linalg::distance::MetricType;

/// Co-located rerank store: an optional per-row vector representation written
/// alongside the index so the refinement step can recompute exact-ish distances
/// without re-reading the wide vector column from the source parquet.
///
/// The IVF-PQ index only persists PQ codes (≈16 B/row); refinement therefore has
/// to fetch the original vectors to re-rank candidates. Reading those from the
/// source parquet page-decodes a multi-MB data page per scattered candidate row
/// — the dominant per-query cost on wide (e.g. dim=1024) embeddings. A rerank
/// store trades build-time storage for a contiguous, page-decode-free refine read.
///
/// This is orthogonal to the coarse index type: it is "store the vector for
/// reranking" independent of how candidates are found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RerankStore {
    /// No rerank store. Refinement reads originals from the source parquet
    /// (`refine_factor > 1`), or is skipped entirely (`refine_factor == 1`,
    /// PQ-approx distances only). This is the default — it adds no storage.
    #[default]
    None,
    /// Scalar-quantized (int8) originals, ≈`dim` bytes/row (4× smaller than raw
    /// f32). Refinement reads a contiguous byte range per candidate and reranks
    /// with integer `l2_u8`. Near-exact recall at a quarter of the f32 footprint.
    /// L2 / Cosine metrics only.
    Sq8,
    /// Full-precision (f32) originals, `dim * 4` bytes/row. Refinement reads a
    /// contiguous byte range per candidate and reranks with exact f32 L2 —
    /// recovering the recall SQ8 rounds away, at 4× SQ8's storage. Same
    /// page-decode-free read path as SQ8; the only difference is precision vs
    /// footprint. L2 / Cosine metrics only.
    Flat,
}

/// Configuration for [`super::ExternalIvfPqIndex::build`].
///
/// Use [`Self::builder`]; defaults match Lance's normal IVF-PQ defaults.
#[derive(Clone, Debug)]
pub struct ExternalIvfPqIndexParams {
    /// Number of IVF partitions (kmeans `k`).
    pub num_partitions: usize,
    /// Number of PQ sub-vectors. Vector dimension must be divisible by this.
    pub num_sub_vectors: usize,
    /// Bits per PQ code. 8 (256 codes) is the universal default.
    pub num_bits_per_sub_vector: usize,
    /// Distance metric.
    pub metric: MetricType,
    /// kmeans iterations during IVF training.
    pub max_iters: usize,
    /// Training sample size = `num_partitions * sample_rate`.
    pub sample_rate: usize,
    /// RNG seed (kmeans + PQ training).
    pub seed: u64,
    /// Optional co-located rerank store. Default [`RerankStore::None`].
    pub rerank_store: RerankStore,
}

impl ExternalIvfPqIndexParams {
    pub fn builder() -> ExternalIvfPqIndexParamsBuilder {
        ExternalIvfPqIndexParamsBuilder::default()
    }
}

#[derive(Clone, Debug)]
pub struct ExternalIvfPqIndexParamsBuilder {
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits_per_sub_vector: usize,
    metric: MetricType,
    max_iters: usize,
    sample_rate: usize,
    seed: u64,
    rerank_store: RerankStore,
}

impl Default for ExternalIvfPqIndexParamsBuilder {
    fn default() -> Self {
        // Defaults match what Lance's dataset-backed IVF-PQ uses today.
        Self {
            num_partitions: 256,
            num_sub_vectors: 16,
            num_bits_per_sub_vector: 8,
            metric: MetricType::L2,
            max_iters: 50,
            sample_rate: 256,
            seed: 0xCAFE_BABE_DEAD_BEEF,
            rerank_store: RerankStore::None,
        }
    }
}

impl ExternalIvfPqIndexParamsBuilder {
    pub fn num_partitions(mut self, n: usize) -> Self {
        self.num_partitions = n;
        self
    }
    pub fn num_sub_vectors(mut self, n: usize) -> Self {
        self.num_sub_vectors = n;
        self
    }
    pub fn num_bits_per_sub_vector(mut self, n: usize) -> Self {
        self.num_bits_per_sub_vector = n;
        self
    }
    pub fn metric(mut self, m: MetricType) -> Self {
        self.metric = m;
        self
    }
    pub fn max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }
    pub fn sample_rate(mut self, n: usize) -> Self {
        self.sample_rate = n;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    /// Enable a co-located rerank store. Default [`RerankStore::None`].
    pub fn rerank_store(mut self, store: RerankStore) -> Self {
        self.rerank_store = store;
        self
    }

    pub fn build(self) -> ExternalIvfPqIndexParams {
        ExternalIvfPqIndexParams {
            num_partitions: self.num_partitions,
            num_sub_vectors: self.num_sub_vectors,
            num_bits_per_sub_vector: self.num_bits_per_sub_vector,
            metric: self.metric,
            max_iters: self.max_iters,
            sample_rate: self.sample_rate,
            seed: self.seed,
            rerank_store: self.rerank_store,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let p = ExternalIvfPqIndexParams::builder().build();
        assert_eq!(p.num_partitions, 256);
        assert_eq!(p.num_sub_vectors, 16);
        assert_eq!(p.num_bits_per_sub_vector, 8);
        assert_eq!(p.metric, MetricType::L2);
    }

    #[test]
    fn builder_overrides() {
        let p = ExternalIvfPqIndexParams::builder()
            .num_partitions(32)
            .num_sub_vectors(4)
            .metric(MetricType::Cosine)
            .max_iters(10)
            .seed(42)
            .build();
        assert_eq!(p.num_partitions, 32);
        assert_eq!(p.num_sub_vectors, 4);
        assert_eq!(p.metric, MetricType::Cosine);
        assert_eq!(p.max_iters, 10);
        assert_eq!(p.seed, 42);
    }
}
