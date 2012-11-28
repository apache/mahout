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

import java.util.Iterator;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class TestSamplingIterator extends MahoutTestCase {

  @Test
  public void testEmptyCase() {
    assertFalse(new SamplingIterator<Integer>(new CountingIterator(0), 0.9999).hasNext());
    assertFalse(new SamplingIterator<Integer>(new CountingIterator(0), 1).hasNext());
  }

  @Test
  public void testSmallInput() {
    Iterator<Integer> t = new SamplingIterator<Integer>(new CountingIterator(1), 0.9999);
    assertTrue(t.hasNext());
    assertEquals(0, t.next().intValue());
    assertFalse(t.hasNext());
  }

  @Test(expected = IllegalArgumentException.class)
  public void testBadRate1() {
    new SamplingIterator<Integer>(new CountingIterator(1), 0.0);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testBadRate2() {
    new SamplingIterator<Integer>(new CountingIterator(1), 1.1);
  }

  @Test
  public void testExactSizeMatch() {
    Iterator<Integer> t = new SamplingIterator<Integer>(new CountingIterator(10), 1);
    for (int i = 0; i < 10; i++) {
      assertTrue(t.hasNext());
      assertEquals(i, t.next().intValue());
    }
    assertFalse(t.hasNext());
  }

  @Test
  public void testSample() {
    for (int i = 0; i < 1000; i++) {
      Iterator<Integer> t = new SamplingIterator<Integer>(new CountingIterator(1000), 0.1);
      int k = 0;
      while (t.hasNext()) {
        int v = t.next();
        k++;
        assertTrue(v >= 0);
        assertTrue(v < 1000);
      }
      double sd = Math.sqrt(0.9 * 0.1 * 1000);
      assertTrue(k >= 100 - 4 * sd);
      assertTrue(k <= 100 + 4 * sd);
    }
  }
}
