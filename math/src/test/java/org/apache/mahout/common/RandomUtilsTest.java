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

package org.apache.mahout.common;

import org.apache.mahout.math.MahoutTestCase;
import org.junit.Test;

import java.util.Random;

/**
 * Tests {@linkRandomUtils}.
 */
public final class RandomUtilsTest extends MahoutTestCase {

  @Test
  public void testIsNotPrime() {
    assertTrue(RandomUtils.isNotPrime(Integer.MIN_VALUE));
    assertTrue(RandomUtils.isNotPrime(-1));
    assertTrue(RandomUtils.isNotPrime(0));
    assertTrue(RandomUtils.isNotPrime(1));
    assertTrue(!RandomUtils.isNotPrime(2));
    assertTrue(!RandomUtils.isNotPrime(3));
    assertTrue(RandomUtils.isNotPrime(4));
    assertTrue(!RandomUtils.isNotPrime(5));
    assertTrue(RandomUtils.isNotPrime(Integer.MAX_VALUE - 1));   
    assertTrue(!RandomUtils.isNotPrime(Integer.MAX_VALUE)); // 2^31 - 1
  }

  @Test
  public void testNextPrime() {
    assertEquals(2, RandomUtils.nextPrime(-1));
    assertEquals(2, RandomUtils.nextPrime(1));
    assertEquals(2, RandomUtils.nextPrime(2));
    assertEquals(3, RandomUtils.nextPrime(3));
    assertEquals(5, RandomUtils.nextPrime(4));
    assertEquals(5, RandomUtils.nextPrime(5));
    assertEquals(7, RandomUtils.nextPrime(6));    
    assertEquals(Integer.MAX_VALUE, RandomUtils.nextPrime(Integer.MAX_VALUE - 1));
  }

  @Test
  public void testNextTwinPrime() {
    assertEquals(5, RandomUtils.nextTwinPrime(-1));
    assertEquals(5, RandomUtils.nextTwinPrime(1));
    assertEquals(5, RandomUtils.nextTwinPrime(2));
    assertEquals(5, RandomUtils.nextTwinPrime(3));
    assertEquals(7, RandomUtils.nextTwinPrime(4));
    assertEquals(7, RandomUtils.nextTwinPrime(5));
    assertEquals(13, RandomUtils.nextTwinPrime(6));
    assertEquals(RandomUtils.MAX_INT_SMALLER_TWIN_PRIME + 2,
                 RandomUtils.nextTwinPrime(RandomUtils.MAX_INT_SMALLER_TWIN_PRIME));
    try {
      RandomUtils.nextTwinPrime(RandomUtils.MAX_INT_SMALLER_TWIN_PRIME + 1);
      fail();
    } catch (IllegalArgumentException iae) {
      // good
    }
  }

  @Test
  public void testLongToSeed() {
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < 10000; i++) {
      long l = r.nextLong();
      byte[] bytes = RandomUtils.longSeedtoBytes(l);
      long back = RandomUtils.seedBytesToLong(bytes);
      assertEquals(l, back);
    }
  }

  @Test
  public void testHashDouble() {
    assertEquals(0, RandomUtils.hashDouble(0.0));
    assertEquals(1072693248, RandomUtils.hashDouble(1.0));
    assertEquals(2146959360, RandomUtils.hashDouble(Double.NaN));
    assertEquals(2146435072, RandomUtils.hashDouble(Double.POSITIVE_INFINITY));
  }

  @Test
  public void testHashFloat() {
    assertEquals(0, RandomUtils.hashFloat(0.0f));
    assertEquals(1065353216, RandomUtils.hashFloat(1.0f));
    assertEquals(2143289344, RandomUtils.hashFloat(Float.NaN));
    assertEquals(2139095040, RandomUtils.hashFloat(Float.POSITIVE_INFINITY));
  }

  @Test
  public void testHashLong() {
    assertEquals(0, RandomUtils.hashLong(-1L));
    assertEquals(0, RandomUtils.hashLong(0L));
    assertEquals(1, RandomUtils.hashLong(1L));
    assertEquals(Integer.MAX_VALUE, RandomUtils.hashLong(Integer.MAX_VALUE));
    assertEquals(Integer.MIN_VALUE, RandomUtils.hashLong((long) Integer.MAX_VALUE + 1L));
    assertEquals(Integer.MIN_VALUE, RandomUtils.hashLong(Long.MAX_VALUE));
    assertEquals(Integer.MIN_VALUE, RandomUtils.hashLong(Long.MIN_VALUE));
  }

}
