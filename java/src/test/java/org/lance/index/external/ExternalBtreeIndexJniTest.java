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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Smoke tests for the scalar (BTree) external index JNI surface. Full end-to-end coverage (build +
 * open + searchKeys + fetchRows) lives in {@code rust/lance/tests/external_scalar_index_phase1.rs},
 * which exercises the same code path the JNI calls into. Java-side seeding of parquet test data
 * requires a parquet writer dependency and is deferred, mirroring {@link
 * ExternalIvfPqIndexJniTest}.
 */
public class ExternalBtreeIndexJniTest {
  static {
    JniLoader.ensureLoaded();
  }

  @Test
  public void paramsBuilderDefaults() {
    ExternalBtreeIndexParams p = ExternalBtreeIndexParams.builder().build();
    assertEquals(ExternalBtreeIndexParams.DEFAULT_BATCH_SIZE, p.getBatchSize());
    assertEquals(4096L, p.getBatchSize());
  }

  @Test
  public void paramsBuilderOverride() {
    ExternalBtreeIndexParams p = ExternalBtreeIndexParams.builder().batchSize(1024L).build();
    assertEquals(1024L, p.getBatchSize());
  }

  @Test
  public void openMissingDirThrows() {
    // Validates the JNI exception bridge for the new nativeOpen. LanceError::IO maps to
    // java.io.IOException.
    assertThrows(
        java.io.IOException.class,
        () -> ExternalBtreeIndex.open("/tmp/this-btree-path-does-not-exist-7c1e4a90"));
  }

  @Test
  public void searchStringKeysReusesPackDeletedRids() {
    // The BTree index reuses ExternalIvfPqIndex.packDeletedRids for the deletedRids argument;
    // confirm the shared packing is reachable from the scalar side of the API.
    java.util.List<long[]> deletes = new java.util.ArrayList<>();
    deletes.add(new long[] {1L, 42L});
    byte[] packed = ExternalIvfPqIndex.packDeletedRids(deletes);
    assertEquals(8, packed.length);
    long rid =
        java.nio.ByteBuffer.wrap(packed, 0, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).getLong();
    assertEquals((1L << 32) | 42L, rid);
  }
}
