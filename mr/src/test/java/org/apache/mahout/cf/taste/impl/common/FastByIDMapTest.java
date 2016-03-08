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
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Map;
import java.util.Random;

/** <p>Tests {@link FastByIDMap}.</p> */
public final class FastByIDMapTest extends TasteTestCase {

  @Test
  public void testPutAndGet() {
    FastByIDMap<Long> map = new FastByIDMap<Long>();
    assertNull(map.get(500000L));
    map.put(500000L, 2L);
    assertEquals(2L, (long) map.get(500000L));
  }
  
  @Test
  public void testRemove() {
    FastByIDMap<Long> map = new FastByIDMap<Long>();
    map.put(500000L, 2L);
    map.remove(500000L);
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    assertNull(map.get(500000L));
  }
  
  @Test
  public void testClear() {
    FastByIDMap<Long> map = new FastByIDMap<Long>();
    map.put(500000L, 2L);
    map.clear();
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    assertNull(map.get(500000L));
  }
  
  @Test
  public void testSizeEmpty() {
    FastByIDMap<Long> map = new FastByIDMap<Long>();
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
    map.put(500000L, 2L);
    assertEquals(1, map.size());
    assertFalse(map.isEmpty());
    map.remove(500000L);
    assertEquals(0, map.size());
    assertTrue(map.isEmpty());
  }
  
  @Test
  public void testContains() {
    FastByIDMap<String> map = buildTestFastMap();
    assertTrue(map.containsKey(500000L));
    assertTrue(map.containsKey(47L));
    assertTrue(map.containsKey(2L));
    assertTrue(map.containsValue("alpha"));
    assertTrue(map.containsValue("bang"));
    assertTrue(map.containsValue("beta"));
    assertFalse(map.containsKey(999));
    assertFalse(map.containsValue("something"));
  }

  @Test
  public void testRehash() {
    FastByIDMap<String> map = buildTestFastMap();
    map.remove(500000L);
    map.rehash();
    assertNull(map.get(500000L));
    assertEquals("bang", map.get(47L));
  }
  
  @Test
  public void testGrow() {
    FastByIDMap<String> map = new FastByIDMap<String>(1,1);
    map.put(500000L, "alpha");
    map.put(47L, "bang");
    assertNull(map.get(500000L));
    assertEquals("bang", map.get(47L));
  }
   
  @Test
  public void testVersusHashMap() {
    FastByIDMap<String> actual = new FastByIDMap<String>();
    Map<Long, String> expected = Maps.newHashMapWithExpectedSize(1000000);
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < 1000000; i++) {
      double d = r.nextDouble();
      Long key = (long) r.nextInt(100);
      if (d < 0.4) {
        assertEquals(expected.get(key), actual.get(key));
      } else {
        if (d < 0.7) {
          assertEquals(expected.put(key, "bang"), actual.put(key, "bang"));
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
    FastByIDMap<String> map = new FastByIDMap<String>();
    map.put(4, "bang");
    assertEquals(1, map.size());
    map.put(47L, "bang");
    assertEquals(2, map.size());
    assertNull(map.get(500000L));
    map.put(47L, "buzz");
    assertEquals(2, map.size());
    assertEquals("buzz", map.get(47L));
  }
  
  
  private static FastByIDMap<String> buildTestFastMap() {
    FastByIDMap<String> map = new FastByIDMap<String>();
    map.put(500000L, "alpha");
    map.put(47L, "bang");
    map.put(2L, "beta");
    return map;
  }
  
}
