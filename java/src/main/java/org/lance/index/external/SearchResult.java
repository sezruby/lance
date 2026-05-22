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

import java.util.Objects;

/**
 * One result from {@link ExternalIvfPqIndex#search}. The result is already refined; the {@code
 * distance} is the exact distance under the index's metric.
 */
public final class SearchResult {
  private final String filePath;
  private final long rowIndex;
  private final float distance;

  public SearchResult(String filePath, long rowIndex, float distance) {
    this.filePath = filePath;
    this.rowIndex = rowIndex;
    this.distance = distance;
  }

  /** Path of the parquet file the row lives in (one of the registered file specs). */
  public String getFilePath() {
    return filePath;
  }

  /** Zero-based row index within {@link #getFilePath()}. */
  public long getRowIndex() {
    return rowIndex;
  }

  /** Exact distance under the index's distance metric. */
  public float getDistance() {
    return distance;
  }

  @Override
  public String toString() {
    return "SearchResult{" + filePath + "@" + rowIndex + " d=" + distance + "}";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof SearchResult)) {
      return false;
    }
    SearchResult that = (SearchResult) o;
    return rowIndex == that.rowIndex
        && Float.compare(distance, that.distance) == 0
        && Objects.equals(filePath, that.filePath);
  }

  @Override
  public int hashCode() {
    return Objects.hash(filePath, rowIndex, distance);
  }
}
