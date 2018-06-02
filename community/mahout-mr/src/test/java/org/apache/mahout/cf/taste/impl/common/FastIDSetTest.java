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

import com.google.common.collect.Sets;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Collection;
import java.util.Random;

/** <p>Tests {@link FastIDSet}.</p> */
public final class FastIDSetTest extends TasteTestCase {

  @Test
  public void testContainsAndAdd() {
    FastIDSet set = new FastIDSet();
    assertFalse(set.contains(1));
    set.add(1);
    assertTrue(set.contains(1));
  }

  @Test
  public void testRemove() {
    FastIDSet set = new FastIDSet();
    set.add(1);
    set.remove(1);
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    assertFalse(set.contains(1));
  }

  @Test
  public void testClear() {
    FastIDSet set = new FastIDSet();
    set.add(1);
    set.clear();
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    assertFalse(set.contains(1));
  }

  @Test
  public void testSizeEmpty() {
    FastIDSet set = new FastIDSet();
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    set.add(1);
    assertEquals(1, set.size());
    assertFalse(set.isEmpty());
    set.remove(1);
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
  }

  @Test
  public void testContains() {
    FastIDSet set = buildTestFastSet();
    assertTrue(set.contains(1));
    assertTrue(set.contains(2));
    assertTrue(set.contains(3));
    assertFalse(set.contains(4));
  }

  @Test
  public void testReservedValues() {
    FastIDSet set = new FastIDSet();
    try {
      set.add(Long.MIN_VALUE);
      fail("Should have thrown IllegalArgumentException");
    } catch (IllegalArgumentException iae) {
      // good
    }
    assertFalse(set.contains(Long.MIN_VALUE));
    try {
      set.add(Long.MAX_VALUE);
      fail("Should have thrown IllegalArgumentException");
    } catch (IllegalArgumentException iae) {
      // good
    }
    assertFalse(set.contains(Long.MAX_VALUE));
  }

  @Test
  public void testRehash() {
    FastIDSet set = buildTestFastSet();
    set.remove(1);
    set.rehash();
    assertFalse(set.contains(1));
  }

  @Test
  public void testGrow() {
    FastIDSet set = new FastIDSet(1);
    set.add(1);
    set.add(2);
    assertTrue(set.contains(1));
    assertTrue(set.contains(2));
  }

  @Test
  public void testIterator() {
    FastIDSet set = buildTestFastSet();
    Collection<Long> expected = Sets.newHashSetWithExpectedSize(3);
    expected.add(1L);
    expected.add(2L);
    expected.add(3L);
    LongPrimitiveIterator it = set.iterator();
    while (it.hasNext()) {
      expected.remove(it.nextLong());
    }
    assertTrue(expected.isEmpty());
  }

  @Test
  public void testVersusHashSet() {
    FastIDSet actual = new FastIDSet(1);
    Collection<Integer> expected = Sets.newHashSetWithExpectedSize(1000000);
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < 1000000; i++) {
      double d = r.nextDouble();
      Integer key = r.nextInt(100);
      if (d < 0.4) {
        assertEquals(expected.contains(key), actual.contains(key));
      } else {
        if (d < 0.7) {
          assertEquals(expected.add(key), actual.add(key));
        } else {
          assertEquals(expected.remove(key), actual.remove(key));
        }
        assertEquals(expected.size(), actual.size());
        assertEquals(expected.isEmpty(), actual.isEmpty());
      }
    }
  }

  private static FastIDSet buildTestFastSet() {
    FastIDSet set = new FastIDSet();
    set.add(1);
    set.add(2);
    set.add(3);
    return set;
  }


}