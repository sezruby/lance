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

import java.util.ArrayList;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Smoke tests for the JNI surface. Full end-to-end coverage (build + open + search + fetchRows)
 * lives in {@code rust/lance/tests/external_index_phase1.rs}, which exercises the same code path
 * the JNI calls into. Java-side seeding of parquet test data requires a parquet writer dependency
 * and is deferred to phase 1.7-followup.
 */
public class ExternalIvfPqIndexJniTest {
  static {
    JniLoader.ensureLoaded();
  }

  @Test
  public void packDeletedRidsLittleEndian() {
    java.util.List<long[]> deletes = new ArrayList<>();
    deletes.add(new long[] {0L, 0L});
    deletes.add(new long[] {1L, 42L});
    byte[] packed = ExternalIvfPqIndex.packDeletedRids(deletes);
    assertEquals(16, packed.length, "two u64s = 16 bytes");

    // First rid: (0 << 32) | 0 = 0
    long first =
        java.nio.ByteBuffer.wrap(packed, 0, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).getLong();
    assertEquals(0L, first);

    // Second rid: (1 << 32) | 42 = 4294967338
    long second =
        java.nio.ByteBuffer.wrap(packed, 8, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).getLong();
    assertEquals((1L << 32) | 42L, second);
  }

  @Test
  public void packDeletedRidsValidatesPairLength() {
    java.util.List<long[]> bad = Collections.singletonList(new long[] {1L, 2L, 3L});
    assertThrows(IllegalArgumentException.class, () -> ExternalIvfPqIndex.packDeletedRids(bad));
  }

  @Test
  public void openMissingDirThrows() {
    // Validates the JNI exception bridge. LanceError::IO maps to java.io.IOException.
    assertThrows(
        java.io.IOException.class,
        () -> ExternalIvfPqIndex.open("/tmp/this-path-does-not-exist-9f3a8e2c"));
  }

  @Test
  public void paramsBuilderDefaults() {
    ExternalIvfPqIndexParams p = ExternalIvfPqIndexParams.builder().build();
    assertEquals(256, p.getNumPartitions());
    assertEquals(16, p.getNumSubVectors());
    assertEquals(8, p.getNumBitsPerSubVector());
    assertEquals(ExternalIvfPqIndexParams.Metric.L2, p.getMetric());
    assertNotNull(p.getMetric().toRustString());
    // Rerank store defaults off and maps to the Rust variant name the JNI parses.
    assertEquals(ExternalIvfPqIndexParams.RerankStore.NONE, p.getRerankStore());
    assertEquals("None", p.getRerankStore().toRustString());
    assertEquals("Sq8", ExternalIvfPqIndexParams.RerankStore.SQ8.toRustString());
    assertEquals("Flat", ExternalIvfPqIndexParams.RerankStore.FLAT.toRustString());
  }

  @Test
  public void paramsBuilderOverrides() {
    ExternalIvfPqIndexParams p =
        ExternalIvfPqIndexParams.builder()
            .numPartitions(32)
            .numSubVectors(4)
            .metric(ExternalIvfPqIndexParams.Metric.Cosine)
            .maxIters(10)
            .seed(42L)
            .build();
    assertEquals(32, p.getNumPartitions());
    assertEquals(4, p.getNumSubVectors());
    assertEquals(ExternalIvfPqIndexParams.Metric.Cosine, p.getMetric());
    assertEquals("Cosine", p.getMetric().toRustString());
    assertEquals(10, p.getMaxIters());
    assertEquals(42L, p.getSeed());
  }
}
