/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

public final class SamplingLongPrimitiveIteratorTest extends TasteTestCase {

  @Test
  public void testEmptyCase() {
    assertFalse(new SamplingLongPrimitiveIterator(
        countingIterator(0), 0.9999).hasNext());
    assertFalse(new SamplingLongPrimitiveIterator(
        countingIterator(0), 1).hasNext());
  }

  @Test
  public void testSmallInput() {
    SamplingLongPrimitiveIterator t = new SamplingLongPrimitiveIterator(
        countingIterator(1), 0.9999);
    assertTrue(t.hasNext());
    assertEquals(0L, t.nextLong());
    assertFalse(t.hasNext());
  }

  @Test(expected = IllegalArgumentException.class)
  public void testBadRate1() {
    new SamplingLongPrimitiveIterator(countingIterator(1), 0.0);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testBadRate2() {
    new SamplingLongPrimitiveIterator(countingIterator(1), 1.1);
  }

  @Test
  public void testExactSizeMatch() {
    SamplingLongPrimitiveIterator t = new SamplingLongPrimitiveIterator(
        countingIterator(10), 1);
    for (int i = 0; i < 10; i++) {
      assertTrue(t.hasNext());
      assertEquals(i, t.next().intValue());
    }
    assertFalse(t.hasNext());
  }

  @Test
  public void testSample() {
    double p = 0.1;
    int n = 1000;
    double sd = Math.sqrt(n * p * (1.0 - p));
    for (int i = 0; i < 1000; i++) {
      SamplingLongPrimitiveIterator t = new SamplingLongPrimitiveIterator(countingIterator(n), p);
      int k = 0;
      while (t.hasNext()) {
        long v = t.nextLong();
        k++;
        assertTrue(v >= 0L);
        assertTrue(v < 1000L);
      }
      // Should be +/- 5 standard deviations except in about 1 out of 1.7M cases
      assertTrue(k >= 100 - 5 * sd);
      assertTrue(k <= 100 + 5 * sd);
    }
  }

  private static LongPrimitiveArrayIterator countingIterator(int to) {
    long[] data = new long[to];
    for (int i = 0; i < to; i++) {
      data[i] = i;
    }
    return new LongPrimitiveArrayIterator(data);
  }

}