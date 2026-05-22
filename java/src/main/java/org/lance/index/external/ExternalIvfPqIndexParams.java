/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.index.external;

/**
 * Build configuration for {@link ExternalIvfPqIndex#build}. Defaults match Lance's IVF-PQ defaults:
 * {@code numPartitions=256}, {@code numSubVectors=16}, {@code numBitsPerSubVector=8}, metric =
 * {@link Metric#L2}.
 */
public final class ExternalIvfPqIndexParams {

  /** Distance metric used by the index. Mirrors lance::Distance. */
  public enum Metric {
    L2,
    Cosine,
    Dot;

    String toRustString() {
      switch (this) {
        case L2:
          return "L2";
        case Cosine:
          return "Cosine";
        case Dot:
          return "Dot";
        default:
          throw new IllegalStateException("unknown metric: " + this);
      }
    }
  }

  private final int numPartitions;
  private final int numSubVectors;
  private final int numBitsPerSubVector;
  private final Metric metric;
  private final int maxIters;
  private final int sampleRate;
  private final long seed;

  private ExternalIvfPqIndexParams(Builder b) {
    this.numPartitions = b.numPartitions;
    this.numSubVectors = b.numSubVectors;
    this.numBitsPerSubVector = b.numBitsPerSubVector;
    this.metric = b.metric;
    this.maxIters = b.maxIters;
    this.sampleRate = b.sampleRate;
    this.seed = b.seed;
  }

  public int getNumPartitions() {
    return numPartitions;
  }

  public int getNumSubVectors() {
    return numSubVectors;
  }

  public int getNumBitsPerSubVector() {
    return numBitsPerSubVector;
  }

  public Metric getMetric() {
    return metric;
  }

  public int getMaxIters() {
    return maxIters;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  public long getSeed() {
    return seed;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private int numPartitions = 256;
    private int numSubVectors = 16;
    private int numBitsPerSubVector = 8;
    private Metric metric = Metric.L2;
    private int maxIters = 50;
    private int sampleRate = 256;
    private long seed = 0xCAFEBABEDEADBEEFL;

    public Builder numPartitions(int n) {
      this.numPartitions = n;
      return this;
    }

    public Builder numSubVectors(int n) {
      this.numSubVectors = n;
      return this;
    }

    public Builder numBitsPerSubVector(int n) {
      this.numBitsPerSubVector = n;
      return this;
    }

    public Builder metric(Metric m) {
      this.metric = m;
      return this;
    }

    public Builder maxIters(int n) {
      this.maxIters = n;
      return this;
    }

    public Builder sampleRate(int n) {
      this.sampleRate = n;
      return this;
    }

    public Builder seed(long seed) {
      this.seed = seed;
      return this;
    }

    public ExternalIvfPqIndexParams build() {
      return new ExternalIvfPqIndexParams(this);
    }
  }
}
