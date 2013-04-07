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

import java.util.Collections;
import java.util.Map;
import java.util.Random;
import java.util.WeakHashMap;

import com.google.common.primitives.Longs;
import org.apache.commons.math3.primes.Primes;

/**
 * <p>
 * The source of random stuff for the whole project. This lets us make all randomness in the project
 * predictable, if desired, for when we run unit tests, which should be repeatable.
 * </p>
 */
public final class RandomUtils {

  /** The largest prime less than 2<sup>31</sup>-1 that is the smaller of a twin prime pair. */
  public static final int MAX_INT_SMALLER_TWIN_PRIME = 2147482949;

  private static final Map<RandomWrapper,Boolean> INSTANCES =
      Collections.synchronizedMap(new WeakHashMap<RandomWrapper,Boolean>());

  private static boolean testSeed = false;

  private RandomUtils() { }
  
  public static void useTestSeed() {
    testSeed = true;
    synchronized (INSTANCES) {
      for (RandomWrapper rng : INSTANCES.keySet()) {
        rng.resetToTestSeed();
      }
    }
  }
  
  public static RandomWrapper getRandom() {
    RandomWrapper random = new RandomWrapper();
    if (testSeed) {
      random.resetToTestSeed();
    }
    INSTANCES.put(random, Boolean.TRUE);
    return random;
  }
  
  public static Random getRandom(long seed) {
    RandomWrapper random = new RandomWrapper(seed);
    INSTANCES.put(random, Boolean.TRUE);
    return random;
  }
  
  /** @return what {@link Double#hashCode()} would return for the same value */
  public static int hashDouble(double value) {
    return Longs.hashCode(Double.doubleToLongBits(value));
  }

  /** @return what {@link Float#hashCode()} would return for the same value */
  public static int hashFloat(float value) {
    return Float.floatToIntBits(value);
  }
  
  /**
   * <p>
   * Finds next-largest "twin primes": numbers p and p+2 such that both are prime. Finds the smallest such p
   * such that the smaller twin, p, is greater than or equal to n. Returns p+2, the larger of the two twins.
   * </p>
   */
  public static int nextTwinPrime(int n) {
    if (n > MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalArgumentException();
    }
    if (n <= 3) {
      return 5;
    }
    int next = Primes.nextPrime(n);
    while (!Primes.isPrime(next + 2)) {
      next = Primes.nextPrime(next + 4);
    }
    return next + 2;
  }
  
}
