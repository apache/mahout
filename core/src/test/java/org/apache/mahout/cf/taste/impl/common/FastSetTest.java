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

import java.util.Collection;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * <p>Tests {@link org.apache.mahout.cf.taste.impl.common.FastSet}.</p>
 */
public final class FastSetTest extends TasteTestCase {

  public void testContainsAndAdd() {
    FastSet<String> set = new FastSet<String>();
    assertFalse(set.contains("foo"));
    set.add("foo");
    assertTrue(set.contains("foo"));
  }


  public void testRemove() {
    FastSet<String> set = new FastSet<String>();
    set.add("foo");
    set.remove("foo");
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    assertFalse(set.contains("foo"));
  }


  public void testClear() {
    FastSet<String> set = new FastSet<String>();
    set.add("foo");
    set.clear();
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    assertFalse(set.contains("foo"));
  }

  public void testSizeEmpty() {
    FastSet<String> set = new FastSet<String>();
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
    set.add("foo");
    assertEquals(1, set.size());
    assertFalse(set.isEmpty());
    set.remove("foo");
    assertEquals(0, set.size());
    assertTrue(set.isEmpty());
  }

  public void testContains() {
    FastSet<String> set = buildTestFastSet();
    assertTrue(set.contains("foo") );
    assertTrue(set.contains("baz"));
    assertTrue(set.contains("alpha"));
    assertFalse(set.contains("something"));
  }

  public void testNull() {
    FastSet<String> set = new FastSet<String>();
    try {
      set.add(null);
      fail("Should have thrown NullPointerException");
    } catch (NullPointerException npe) {
      // good
    }
    assertFalse(set.contains(null));
  }

  public void testRehash() {
    FastSet<String> set = buildTestFastSet();
    set.remove("foo");
    set.rehash();
    assertFalse(set.contains("foo"));
  }

  public void testGrow() {
    FastMap<String, String> map = new FastMap<String, String>(1, FastMap.NO_MAX_SIZE);
    map.put("foo", "bar");
    map.put("baz", "bang");
    assertEquals("bar", map.get("foo"));
    assertEquals("bang", map.get("baz"));
  }

  public void testIterator() {
    FastSet<String> set = buildTestFastSet();
    Collection<String> expected = new HashSet<String>(3);
    expected.add("foo");
    expected.add("baz");
    expected.add("alpha");
    for (String s : set) {
      expected.remove(s);
    }
    assertTrue(expected.isEmpty());
  }


  public void testVersusHashSet() {
    Set<Integer> actual = new FastSet<Integer>(1);
    Set<Integer> expected = new HashSet<Integer>(1000000);
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

  private static FastSet<String> buildTestFastSet() {
    FastSet<String> set = new FastSet<String>();
    set.add("foo");
    set.add("baz");
    set.add("alpha");
    return set;
  }


}