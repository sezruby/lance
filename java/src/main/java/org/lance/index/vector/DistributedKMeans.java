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
package org.lance.index.vector;

import org.lance.JniLoader;
import org.lance.index.DistanceType;

import java.util.Objects;

/**
 * Scheduler-neutral distributed IVF centroid-training primitives.
 *
 * <p>Mirrors {@code lance_index::vector::kmeans::distributed}. Callers (Spark, custom RPC) own
 * broadcast, tree-reduce, and convergence; this class exposes only the math — one partial-aggregate
 * Lloyd iteration decomposed into an executor E-step (compute per-cluster stats), a driver/executor
 * reduce (fold the stats), and a driver M-step ({@link #finalizeCentroids}). The E-step over the
 * external parquet index's cached sample lives on {@code
 * org.lance.index.external.ExternalIvfPqIndex#computePartialStatsResident} (which reads parquet
 * rather than a Lance {@code Dataset}); the reduce / M-step / init helpers here are shared across
 * build paths.
 *
 * <p>Every payload that crosses the JNI boundary — partial stats and centroid arrays — is opaque
 * Arrow IPC {@code byte[]}, so callers plumb them straight through a Spark {@code treeReduce} /
 * {@code broadcast} without needing a {@link org.apache.arrow.memory.BufferAllocator} or decoding
 * the vectors. The centroid-producing helpers ({@link #finalizeCentroids}, {@link
 * #selectInitialCentroids}, {@link #bootstrapCentroids}) emit a single-column {@code FixedSizeList}
 * IPC batch whose inner element dtype (Float16/Float32/Float64) is preserved end-to-end.
 */
public final class DistributedKMeans {

  static {
    JniLoader.ensureLoaded();
  }

  private DistributedKMeans() {}

  /** Combine two partial-stats IPC batches into one (associative + commutative). */
  public static byte[] mergePartialStats(byte[] a, byte[] b) {
    Objects.requireNonNull(a, "a");
    Objects.requireNonNull(b, "b");
    return nativeMergePartialStats(a, b);
  }

  /** Fold a batch of partial-stats IPC batches into one (driver-side tree-reduce sink). */
  public static byte[] reducePartialStats(byte[][] stats) {
    Objects.requireNonNull(stats, "stats");
    return nativeReducePartialStats(stats);
  }

  /**
   * M-step: compute new centroids from the reduced {@code stats}, carrying forward the previous
   * centroid for any cluster that drew no points this round. Returns a single-column {@code
   * FixedSizeList} IPC batch whose inner dtype matches {@code prevCentroids}.
   */
  public static byte[] finalizeCentroids(byte[] stats, byte[] prevCentroids) {
    Objects.requireNonNull(stats, "stats");
    Objects.requireNonNull(prevCentroids, "prevCentroids");
    return nativeFinalizeCentroids(stats, prevCentroids);
  }

  /**
   * Driver-side init: pick {@code k} rows uniformly at random from the worker sample chunks.
   * Returns a single-column {@code FixedSizeList} IPC batch whose inner dtype matches the samples.
   */
  public static byte[] selectInitialCentroids(byte[][] sampleChunks, int k, long rngSeed) {
    Objects.requireNonNull(sampleChunks, "sampleChunks");
    return nativeSelectInitialCentroids(sampleChunks, k, rngSeed);
  }

  /**
   * Driver-side init: bootstrap centroids by running single-machine kmeans over the worker sample
   * chunks. Higher-quality than {@link #selectInitialCentroids} but O(sample); prefer random init
   * when the collected sample is large. Returns a single-column {@code FixedSizeList} IPC batch
   * whose inner dtype matches the samples.
   */
  public static byte[] bootstrapCentroids(
      byte[][] sampleChunks, int k, DistanceType distanceType, long rngSeed) {
    Objects.requireNonNull(sampleChunks, "sampleChunks");
    Objects.requireNonNull(distanceType, "distanceType");
    return nativeBootstrapCentroids(sampleChunks, k, distanceType.toString(), rngSeed);
  }

  // -- native ---------------------------------------------------------------

  private static native byte[] nativeMergePartialStats(byte[] a, byte[] b);

  private static native byte[] nativeReducePartialStats(byte[][] stats);

  private static native byte[] nativeFinalizeCentroids(byte[] stats, byte[] prevCentroids);

  private static native byte[] nativeSelectInitialCentroids(
      byte[][] sampleChunks, int k, long rngSeed);

  private static native byte[] nativeBootstrapCentroids(
      byte[][] sampleChunks, int k, String distanceType, long rngSeed);
}
