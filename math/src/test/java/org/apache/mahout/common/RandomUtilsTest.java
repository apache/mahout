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
  public void testHashDouble() {
    assertEquals(new Double(0.0).hashCode(), RandomUtils.hashDouble(0.0));
    assertEquals(new Double(1.0).hashCode(), RandomUtils.hashDouble(1.0));
    assertEquals(new Double(Double.POSITIVE_INFINITY).hashCode(), RandomUtils.hashDouble(Double.POSITIVE_INFINITY));
    assertEquals(new Double(Double.NaN).hashCode(), RandomUtils.hashDouble(Double.NaN));
  }

  @Test
  public void testHashFloat() {
    assertEquals(new Float(0.0f).hashCode(), RandomUtils.hashFloat(0.0f));
    assertEquals(new Float(1.0f).hashCode(), RandomUtils.hashFloat(1.0f));
    assertEquals(new Float(Float.POSITIVE_INFINITY).hashCode(), RandomUtils.hashFloat(Float.POSITIVE_INFINITY));
    assertEquals(new Float(Float.NaN).hashCode(), RandomUtils.hashFloat(Float.NaN));
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
  public void testSetSeed() {
    Random rTest0 = RandomUtils.getRandom();
    Random rTest1 = RandomUtils.getRandom();
    Random r0 = RandomUtils.getRandom(0);
    Random r1 = RandomUtils.getRandom(1);

    long lTest0 = rTest0.nextLong();
    long lTest1 = rTest1.nextLong();
    long l0 = r0.nextLong();
    long l1 = r1.nextLong();
    assertEquals("getRandom() must match getRandom() in unit tests", lTest0, lTest1);
    assertTrue("getRandom() must differ from getRandom(0)", lTest0 != l1);
    assertTrue("getRandom(0) must differ from getRandom(1)", l0 != l1);
  }

}
