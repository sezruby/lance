// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build parameters for [`super::ExternalIvfPqIndex`].

use lance_linalg::distance::MetricType;

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

    pub fn build(self) -> ExternalIvfPqIndexParams {
        ExternalIvfPqIndexParams {
            num_partitions: self.num_partitions,
            num_sub_vectors: self.num_sub_vectors,
            num_bits_per_sub_vector: self.num_bits_per_sub_vector,
            metric: self.metric,
            max_iters: self.max_iters,
            sample_rate: self.sample_rate,
            seed: self.seed,
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
