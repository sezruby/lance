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

import java.util.Arrays;
import java.util.List;

/**
 * Java handle for a scalar (BTree) index built over caller-registered parquet files — the scalar
 * analog of {@link ExternalIvfPqIndex}.
 *
 * <p>Lance builds and queries a BTree over the parquet data in place, without ingesting it into a
 * Lance dataset. Callers register a list of parquet files keyed on a scalar column; lookups return
 * the matched-target-identity SET as {@link SearchResult} {@code (filePath, rowIndex)} (with {@code
 * distance = 0.0}, meaningless for a point lookup). Callers never decode an internal rid.
 *
 * <p>This powers an "indexed Delta MERGE": for each source merge key, find the matching target rows
 * via a BTree over the target's merge key. The returned {@code (filePath, rowIndex)} SET is exactly
 * what Delta {@code findTouchedFiles} needs.
 *
 * <p>Lifecycle: {@link #build} writes an index directory under an output URI, returning a UUID that
 * names the directory; {@link #open} returns an open handle. {@link #close} releases it. The handle
 * is {@link AutoCloseable}; use try-with-resources where possible.
 *
 * <p>Example:
 *
 * <pre>{@code
 * String uuid = ExternalBtreeIndex.build(
 *     List.of("/data/target-0.parquet", "/data/target-1.parquet"),
 *     "id",
 *     "/index/v1",
 *     ExternalBtreeIndexParams.builder().build());
 *
 * try (ExternalBtreeIndex idx = ExternalBtreeIndex.open("/index/v1/" + uuid)) {
 *   List<SearchResult> hits = idx.searchLongKeys(new long[] {1L, 2L, 42L}, null);
 *   byte[] arrowIpc = idx.fetchRows(
 *       hits.stream().map(h -> ParquetRowKey.of(h.getFilePath(), h.getRowIndex())).toList(),
 *       List.of("id", "payload"));
 * }
 * }</pre>
 */
public final class ExternalBtreeIndex implements AutoCloseable {

  static {
    JniLoader.ensureLoaded();
  }

  private static final String[] EMPTY_STRINGS = new String[0];
  private static final long[] EMPTY_LONGS = new long[0];

  private long handle;

  private ExternalBtreeIndex(long handle) {
    this.handle = handle;
  }

  // ---- build / open ---------------------------------------------------------

  /**
   * Build an external BTree index over the registered parquet files, keyed on {@code keyColumn}.
   *
   * <p>Writes a single index directory under {@code outputUri} named by the returned UUID
   * (containing the BTree files + {@code manifest.json}). {@code keyColumn} must be a scalar
   * (non-nested) column present in every file's schema with the same Arrow type. The {@code
   * file_id} encoded into rids is implicit in {@code filePaths}'s position; reordering invalidates
   * the index.
   *
   * @return UUID directory name (not the full URI); join with {@code outputUri} to get the open
   *     URI.
   */
  public static String build(
      List<String> filePaths, String keyColumn, String outputUri, ExternalBtreeIndexParams params) {
    return nativeBuild(
        filePaths.toArray(new String[0]), keyColumn, outputUri, params.getBatchSize());
  }

  /**
   * Open an external BTree index by URI. Cheap: reads {@code manifest.json} and the BTree footer.
   */
  public static ExternalBtreeIndex open(String uri) {
    return new ExternalBtreeIndex(nativeOpen(uri));
  }

  // ---- search ---------------------------------------------------------------

  /**
   * Look up all rows whose {@code int64} key equals one of {@code keys}.
   *
   * <p>Runs a single BTree {@code IsIn} query and returns the matched-target-identity SET (the
   * union across keys, not per-key association). Use this when the index's key column is a 64-bit
   * integer.
   *
   * @param keys the source merge keys to look up; batch as many as possible into one call.
   * @param deletedRids optional little-endian {@code u64} packed array of deleted rids encoded as
   *     {@code (file_id << 32) | row_index}. Matches will exclude these. {@code null} means no
   *     filter. Build it with {@link ExternalIvfPqIndex#packDeletedRids} (e.g. from a Delta
   *     deletion vector).
   */
  public List<SearchResult> searchLongKeys(long[] keys, byte[] deletedRids) {
    return Arrays.asList(nativeSearchKeys(handle, "int64", keys, EMPTY_STRINGS, deletedRids));
  }

  /**
   * Look up all rows whose {@code utf8} (string) key equals one of {@code keys}. Same {@code IsIn}
   * SET semantics as {@link #searchLongKeys}; use this when the index's key column is a string.
   *
   * @param keys the source merge keys to look up; batch as many as possible into one call.
   * @param deletedRids optional packed deleted rids; see {@link #searchLongKeys}.
   */
  public List<SearchResult> searchStringKeys(String[] keys, byte[] deletedRids) {
    return Arrays.asList(nativeSearchKeys(handle, "utf8", EMPTY_LONGS, keys, deletedRids));
  }

  // ---- fetch ----------------------------------------------------------------

  /**
   * Fetch arbitrary projection columns for {@code rowKeys} from the registered parquet files.
   *
   * <p>Returns Arrow IPC stream bytes; decode with {@code ArrowStreamReader} on the Java side. The
   * batch has one row per input key, in caller-input order. {@code projection} may include any
   * parquet column, not just the key column.
   */
  public byte[] fetchRows(List<ParquetRowKey> rowKeys, List<String> projection) {
    String[] paths = new String[rowKeys.size()];
    long[] rows = new long[rowKeys.size()];
    for (int i = 0; i < rowKeys.size(); i++) {
      paths[i] = rowKeys.get(i).getFilePath();
      rows[i] = rowKeys.get(i).getRowIndex();
    }
    return nativeFetchRows(handle, paths, rows, projection.toArray(new String[0]));
  }

  // ---- accessors ------------------------------------------------------------

  /** Number of registered parquet files. */
  public int getNumFiles() {
    return nativeNumFiles(handle);
  }

  /** Scalar key column the index was built over. */
  public String getKeyColumn() {
    return nativeKeyColumn(handle);
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
      String[] filePaths, String keyColumn, String outputUri, long batchSize);

  private static native long nativeOpen(String uri);

  private static native void nativeClose(long handle);

  private static native SearchResult[] nativeSearchKeys(
      long handle, String keyType, long[] longKeys, String[] stringKeys, byte[] deletedRids);

  private static native byte[] nativeFetchRows(
      long handle, String[] filePaths, long[] rowIndices, String[] projection);

  private static native int nativeNumFiles(long handle);

  private static native String nativeKeyColumn(long handle);
}
