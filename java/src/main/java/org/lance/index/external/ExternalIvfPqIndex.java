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

import org.lance.JniLoader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Java handle for an IVF-PQ index built over caller-registered parquet files.
 *
 * <p>Lance owns the parquet reader and refinement; callers see {@link SearchResult} as {@code
 * (filePath, rowIndex, distance)} and never have to decode an internal rid.
 *
 * <p>Lifecycle: {@link #build} writes an index file under an output URI, returning a UUID that
 * names the index directory; {@link #open} returns an open handle. {@link #close} releases the
 * handle. The handle is {@link AutoCloseable}; use try-with-resources where possible.
 *
 * <p>Example:
 *
 * <pre>{@code
 * String uuid = ExternalIvfPqIndex.build(
 *     List.of("/data/embeddings-0.parquet", "/data/embeddings-1.parquet"),
 *     "vec",
 *     "/index/v1",
 *     ExternalIvfPqIndexParams.builder().numPartitions(256).numSubVectors(16).build());
 *
 * try (ExternalIvfPqIndex idx = ExternalIvfPqIndex.open("/index/v1/" + uuid)) {
 *   List<SearchResult> hits = idx.search(query, 10, 16, 8, null);
 *   byte[] arrowIpc = idx.fetchRows(
 *       hits.stream().map(h -> ParquetRowKey.of(h.getFilePath(), h.getRowIndex())).toList(),
 *       List.of("doc_id", "title"));
 * }
 * }</pre>
 */
public final class ExternalIvfPqIndex implements AutoCloseable {

  static {
    JniLoader.ensureLoaded();
  }

  private long handle;

  private ExternalIvfPqIndex(long handle) {
    this.handle = handle;
  }

  // ---- build / open ---------------------------------------------------------

  /**
   * Build an external IVF-PQ index over the registered parquet files.
   *
   * <p>Writes a single index directory under {@code outputUri} named by the returned UUID. The
   * directory contains {@code index.idx} (IVF model + PQ codebooks) and {@code manifest.json}
   * (parquet file list + build params).
   *
   * <p>The {@code file_id} encoded into rids is implicit in {@code filePaths}'s position;
   * reordering invalidates the index.
   *
   * @return UUID directory name (not the full URI); join with {@code outputUri} to get the open
   *     URI.
   */
  public static String build(
      List<String> filePaths,
      String vectorColumn,
      String outputUri,
      ExternalIvfPqIndexParams params) {
    String[] paths = filePaths.toArray(new String[0]);
    return nativeBuild(
        paths,
        vectorColumn,
        outputUri,
        params.getNumPartitions(),
        params.getNumSubVectors(),
        params.getNumBitsPerSubVector(),
        params.getMetric().toRustString(),
        params.getMaxIters(),
        params.getSampleRate(),
        params.getSeed());
  }

  /**
   * Open an external IVF-PQ index by URI. Cheap: reads {@code manifest.json} and the index file
   * footer.
   */
  public static ExternalIvfPqIndex open(String uri) {
    long handle = nativeOpen(uri);
    return new ExternalIvfPqIndex(handle);
  }

  /**
   * Run a vector query and return up to {@code k} refined results.
   *
   * @param query Query vector. Length must match the index's dimension.
   * @param k Number of results to return.
   * @param nprobes Number of IVF partitions to probe.
   * @param refineFactor Re-rank multiplier; {@code k * refineFactor} approximate candidates are
   *     fetched, refined exactly, then trimmed to {@code k}.
   * @param deletedRids Optional little-endian {@code u64} packed array of deleted rids encoded as
   *     {@code (file_id << 32) | row_index}. Survivors will exclude these. {@code null} means no
   *     filter.
   */
  public List<SearchResult> search(
      float[] query, int k, int nprobes, int refineFactor, byte[] deletedRids) {
    SearchResult[] arr = nativeSearch(handle, query, k, nprobes, refineFactor, deletedRids);
    return java.util.Arrays.asList(arr);
  }

  /**
   * Fetch arbitrary projection columns for {@code rowKeys} from the registered parquet files.
   *
   * <p>Returns Arrow IPC stream bytes; decode with {@code ArrowStreamReader} on the Java side. The
   * batch has one row per input key, in caller-input order.
   *
   * <p>Use this for post-topK materialization: pass the {@code (filePath, rowIndex)} pairs from
   * {@link #search} and project only the columns you need.
   */
  public byte[] fetchRows(List<ParquetRowKey> rowKeys, List<String> projection) {
    String[] paths = new String[rowKeys.size()];
    long[] rows = new long[rowKeys.size()];
    for (int i = 0; i < rowKeys.size(); i++) {
      paths[i] = rowKeys.get(i).getFilePath();
      rows[i] = rowKeys.get(i).getRowIndex();
    }
    String[] proj = projection.toArray(new String[0]);
    return nativeFetchRows(handle, paths, rows, proj);
  }

  /** Pack a list of {@code (file_id, row_index)} deletes into the byte format {@link #search}. */
  public static byte[] packDeletedRids(List<long[]> deletedFileRowPairs) {
    ByteBuffer buf =
        ByteBuffer.allocate(deletedFileRowPairs.size() * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (long[] pair : deletedFileRowPairs) {
      if (pair.length != 2) {
        throw new IllegalArgumentException(
            "expected (file_id, row_index) pair; got length " + pair.length);
      }
      long rid = (pair[0] << 32) | (pair[1] & 0xFFFFFFFFL);
      buf.putLong(rid);
    }
    return buf.array();
  }

  /** Number of IVF partitions. */
  public int getNumPartitions() {
    return nativeNumPartitions(handle);
  }

  /** Number of registered parquet files. */
  public int getNumFiles() {
    return nativeNumFiles(handle);
  }

  /** Vector column the index was built over. */
  public String getVectorColumn() {
    return nativeVectorColumn(handle);
  }

  @Override
  public void close() {
    if (handle != 0L) {
      nativeClose(handle);
      handle = 0L;
    }
  }

  // ---- native methods -------------------------------------------------------

  private static native String nativeBuild(
      String[] filePaths,
      String vectorColumn,
      String outputUri,
      int numPartitions,
      int numSubVectors,
      int numBitsPerSubVector,
      String metric,
      int maxIters,
      int sampleRate,
      long seed);

  private static native long nativeOpen(String uri);

  private static native void nativeClose(long handle);

  private static native SearchResult[] nativeSearch(
      long handle, float[] query, int k, int nprobes, int refineFactor, byte[] deletedRids);

  private static native byte[] nativeFetchRows(
      long handle, String[] filePaths, long[] rowIndices, String[] projection);

  private static native int nativeNumPartitions(long handle);

  private static native int nativeNumFiles(long handle);

  private static native String nativeVectorColumn(long handle);
}
