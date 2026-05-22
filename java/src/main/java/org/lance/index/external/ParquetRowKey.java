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

/** {@code (file_path, row_index)} input to {@link ExternalIvfPqIndex#fetchRows}. */
public final class ParquetRowKey {
  private final String filePath;
  private final long rowIndex;

  public ParquetRowKey(String filePath, long rowIndex) {
    this.filePath = filePath;
    this.rowIndex = rowIndex;
  }

  public static ParquetRowKey of(String filePath, long rowIndex) {
    return new ParquetRowKey(filePath, rowIndex);
  }

  public String getFilePath() {
    return filePath;
  }

  public long getRowIndex() {
    return rowIndex;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ParquetRowKey)) {
      return false;
    }
    ParquetRowKey that = (ParquetRowKey) o;
    return rowIndex == that.rowIndex && Objects.equals(filePath, that.filePath);
  }

  @Override
  public int hashCode() {
    return Objects.hash(filePath, rowIndex);
  }
}
