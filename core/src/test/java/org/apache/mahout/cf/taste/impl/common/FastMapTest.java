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

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/** <p>Tests {@link FastMap}.</p> */
public final class FastMapTest extends TasteTestCase {

  @Test
  public void testPutAndGet() {
    Map<String, String> map = new FastMap<String, String>();
    assertNull(map.get("foo"));
    map.put("foo", "bar");
    assertEquals("bar", map.get("foo"));
  }

  @Test
  public void testRemove() {
    Map<String, String> map = new FastMap<String, String>();
    map.put("foo", "bar");
    map.remove("foo");
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    assertNull(map.get("foo"));
  }

  @Test
  public void testClear() {
    Map<String, String> map = new FastMap<String, String>();
    map.put("foo", "bar");
    map.clear();
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    assertNull(map.get("foo"));
  }

  @Test
  public void testSizeEmpty() {
    Map<String, String> map = new FastMap<String, String>();
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    map.put("foo", "bar");
    assertEquals(1, map.size());
    assertFalse(map.isEmpty());
    map.remove("foo");
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
  }

  @Test
  public void testContains() {
    FastMap<String, String> map = buildTestFastMap();
    assertTrue(map.containsKey("foo"));
    assertTrue(map.containsKey("baz"));
    assertTrue(map.containsKey("alpha"));
    assertTrue(map.containsValue("bar"));
    assertTrue(map.containsValue("bang"));
    assertTrue(map.containsValue("beta"));
    assertFalse(map.containsKey("something"));
    assertFalse(map.containsValue("something"));
  }

  @Test(expected = NullPointerException.class)
  public void testNull1() {
    Map<String, String> map = new FastMap<String, String>();
    assertNull(map.get(null));
    map.put(null, "bar");
  }

  @Test(expected = NullPointerException.class)
  public void testNull2() {
    Map<String, String> map = new FastMap<String, String>();
    map.put("foo", null);
  }

  @Test
  public void testRehash() {
    FastMap<String, String> map = buildTestFastMap();
    map.remove("foo");
    map.rehash();
    assertNull(map.get("foo"));
    assertEquals("bang", map.get("baz"));
  }

  @Test
  public void testGrow() {
    Map<String, String> map = new FastMap<String, String>(1, FastMap.NO_MAX_SIZE);
    map.put("foo", "bar");
    map.put("baz", "bang");
    assertEquals("bar", map.get("foo"));
    assertEquals("bang", map.get("baz"));
  }

  @Test
  public void testKeySet() {
    FastMap<String, String> map = buildTestFastMap();
    Collection<String> expected = Sets.newHashSetWithExpectedSize(3);
    expected.add("foo");
    expected.add("baz");
    expected.add("alpha");
    Set<String> actual = map.keySet();
    assertTrue(expected.containsAll(actual));
    assertTrue(actual.containsAll(expected));
    Iterator<String> it = actual.iterator();
    while (it.hasNext()) {
      String value = it.next();
      if (!"baz".equals(value)) {
        it.remove();
      }
    }
    assertTrue(map.containsKey("baz"));
    assertFalse(map.containsKey("foo"));
    assertFalse(map.containsKey("alpha"));
  }

  @Test
  public void testValues() {
    FastMap<String, String> map = buildTestFastMap();
    Collection<String> expected = Sets.newHashSetWithExpectedSize(3);
    expected.add("bar");
    expected.add("bang");
    expected.add("beta");
    Collection<String> actual = map.values();
    assertTrue(expected.containsAll(actual));
    assertTrue(actual.containsAll(expected));
    Iterator<String> it = actual.iterator();
    while (it.hasNext()) {
      String value = it.next();
      if (!"bang".equals(value)) {
        it.remove();
      }
    }
    assertTrue(map.containsValue("bang"));
    assertFalse(map.containsValue("bar"));
    assertFalse(map.containsValue("beta"));
  }

  @Test
  public void testEntrySet() {
    FastMap<String, String> map = buildTestFastMap();
    Set<Map.Entry<String, String>> actual = map.entrySet();
    Collection<String> expectedKeys = Sets.newHashSetWithExpectedSize(3);
    expectedKeys.add("foo");
    expectedKeys.add("baz");
    expectedKeys.add("alpha");
    Collection<String> expectedValues = Sets.newHashSetWithExpectedSize(3);
    expectedValues.add("bar");
    expectedValues.add("bang");
    expectedValues.add("beta");
    assertEquals(3, actual.size());
    for (Map.Entry<String, String> entry : actual) {
      expectedKeys.remove(entry.getKey());
      expectedValues.remove(entry.getValue());
    }
    assertEquals(0, expectedKeys.size());
    assertEquals(0, expectedValues.size());
  }

  @Test
  public void testVersusHashMap() {
    Map<Integer, String> actual = new FastMap<Integer, String>(1, 1000000);
    Map<Integer, String> expected = Maps.newHashMapWithExpectedSize(1000000);
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < 1000000; i++) {
      double d = r.nextDouble();
      Integer key = r.nextInt(100);
      if (d < 0.4) {
        assertEquals(expected.get(key), actual.get(key));
      } else {
        if (d < 0.7) {
          assertEquals(expected.put(key, "foo"), actual.put(key, "foo"));
        } else {
          assertEquals(expected.remove(key), actual.remove(key));
        }
        assertEquals(expected.size(), actual.size());
        assertEquals(expected.isEmpty(), actual.isEmpty());
      }
    }
  }

  @Test
  public void testMaxSize() {
    Map<String, String> map = new FastMap<String, String>(1, 1);
    map.put("foo", "bar");
    assertEquals(1, map.size());
    map.put("baz", "bang");
    assertEquals(1, map.size());
    assertNull(map.get("foo"));
    map.put("baz", "buzz");
    assertEquals(1, map.size());
    assertEquals("buzz", map.get("baz"));
  }

  private static FastMap<String, String> buildTestFastMap() {
    FastMap<String, String> map = new FastMap<String, String>();
    map.put("foo", "bar");
    map.put("baz", "bang");
    map.put("alpha", "beta");
    return map;
  }

}
