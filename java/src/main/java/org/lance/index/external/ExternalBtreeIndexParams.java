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
 * Build configuration for {@link ExternalBtreeIndex#build}.
 *
 * <p>A BTree over a scalar key has far fewer tunables than IVF-PQ, so this only carries the page
 * size. Mirrors {@code lance::index::scalar_external::ExternalBtreeIndexParams}.
 */
public final class ExternalBtreeIndexParams {

  /**
   * Default BTree page size (rows per page). Matches Lance's dataset-backed BTree default and the
   * Rust {@code DEFAULT_BTREE_BATCH_SIZE}.
   */
  public static final long DEFAULT_BATCH_SIZE = 4096L;

  private final long batchSize;

  private ExternalBtreeIndexParams(Builder b) {
    this.batchSize = b.batchSize;
  }

  /**
   * Rows per BTree page. Larger pages mean fewer, coarser pages (cheaper lookup metadata, more
   * per-page scan); smaller pages mean finer pruning.
   */
  public long getBatchSize() {
    return batchSize;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private long batchSize = DEFAULT_BATCH_SIZE;

    /** Override the BTree page size. Must be positive. Default {@link #DEFAULT_BATCH_SIZE}. */
    public Builder batchSize(long n) {
      this.batchSize = n;
      return this;
    }

    public ExternalBtreeIndexParams build() {
      return new ExternalBtreeIndexParams(this);
    }
  }
}
