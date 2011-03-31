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

package org.apache.mahout.common.iterator;

import java.util.Collections;
import java.util.Iterator;
import java.util.Arrays;
import java.util.List;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public abstract class SamplerCase extends MahoutTestCase {
  // these provide access to the underlying implementation
  protected abstract Iterator<Integer> createSampler(int n, Iterator<Integer> source);

  protected abstract boolean isSorted();

  @Test
  public void testEmptyCase() {
    assertFalse(createSampler(100, Integers.iterator(0)).hasNext());
  }

  @Test
  public void testSmallInput() {
    Iterator<Integer> t = createSampler(10, Integers.iterator(1));
    assertTrue(t.hasNext());
    assertEquals(0, t.next().intValue());
    assertFalse(t.hasNext());

    t = createSampler(10, Integers.iterator(1));
    assertTrue(t.hasNext());
    assertEquals(0, t.next().intValue());
    assertFalse(t.hasNext());
  }

  @Test
  public void testAbsurdSize() {
    Iterator<Integer> t = createSampler(0, Integers.iterator(2));
    assertFalse(t.hasNext());
  }

  @Test
  public void testExactSizeMatch() {
    Iterator<Integer> t = createSampler(10, Integers.iterator(10));
    for (int i = 0; i < 10; i++) {
      assertTrue(t.hasNext());
      assertEquals(i, t.next().intValue());
    }
    assertFalse(t.hasNext());
  }

  @Test
  public void testSample() {
    Iterator<Integer> source = Integers.iterator(100);
    Iterator<Integer> t = createSampler(15, source);

    // this is just a regression test, not a real test
    List<Integer> expectedValues = Arrays.asList(16, 23, 2, 3, 32, 85, 6, 53, 8, 75, 15, 81, 12, 59, 14);
    if (isSorted()) {
      Collections.sort(expectedValues);
    }
    Iterator<Integer> expected = expectedValues.iterator();
    int last = Integer.MIN_VALUE;
    for (int i = 0; i < 15; i++) {
      assertTrue(t.hasNext());
      int actual = t.next();
      if (isSorted()) {
        assertTrue(actual >= last);
        last = actual;
      } else {
        // any of the first few values should be in the original places
        if (actual < 15) {
          assertEquals(i, actual);
        }
      }

      assertTrue(actual >= 0 && actual < 100);

      // this is just a regression test, but still of some value
      assertEquals(expected.next().intValue(), actual);
      assertFalse(source.hasNext());
    }
    assertFalse(t.hasNext());
  }
}
