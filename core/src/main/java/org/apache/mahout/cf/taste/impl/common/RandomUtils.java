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

import java.util.Random;

/**
 * <p>The source of random stuff for the whole project. This lets us make all randomness in the project predictable, if
 * desired, for when we run unit tests, which should be repeatable.</p>
 *
 * <p>This class is increasingly incorrectly named as it also includes other mathematical utility methods.</p>
 */
public final class RandomUtils {

  private static final long STANDARD_SEED = 0xCAFEBABECAFEBABEL;
  private static boolean testSeed;

  /** The largest prime less than 2<sup>31</sup>-1 that is the smaller of a twin prime pair. */
  public static final int MAX_INT_SMALLER_TWIN_PRIME = 2147482949;

  private RandomUtils() {
  }

  public static void useTestSeed() {
    testSeed = true;
  }

  public static Random getRandom() {
    return testSeed ? new Random(STANDARD_SEED) : new Random();
  }

  /** @return what {@link Double#hashCode()} would return for the same value */
  public static int hashDouble(double value) {
    // Just copied from Double.hashCode
    long bits = Double.doubleToLongBits(value);
    return (int) (bits ^ (bits >>> 32));
  }

  public static int hashFloat(float value) {
    return Float.floatToIntBits(value);
  }

  public static int hashLong(long value) {
    return (int) (value ^ (value >>> 32));
  }

  /**
   * <p>Finds next-largest "twin primes": numbers p and p+2 such that both are prime. Finds the smallest such p such
   * that the smaller twin, p, is greater than or equal to n. Returns p+2, the larger of the two twins.</p>
   */
  public static int nextTwinPrime(int n) {
    if (n > MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalArgumentException();
    }
    if (n <= 3) {
      return 3;
    }
    int next = nextPrime(n);
    while (isNotPrime(next + 2)) {
      next = nextPrime(next + 4);
    }
    return next + 2;
  }

  /** <p>Finds smallest prime p such that p is greater than or equal to n.</p> */
  public static int nextPrime(int n) {
    if (n < 2) {
      return 2;
    }
    // Make sure the number is odd. Is this too clever?
    n |= 0x1;
    // There is no problem with overflow since Integer.MAX_INT is prime, as it happens
    while (isNotPrime(n)) {
      n += 2;
    }
    return n;
  }

  /** @return <code>true</code> iff n is not a prime */
  public static boolean isNotPrime(int n) {
    if (n < 2 || (n & 0x1) == 0) { // < 2 or even
      return true;
    }
    int max = 1 + (int) Math.sqrt((double) n);
    for (int d = 3; d <= max; d += 2) {
      if (n % d == 0) {
        return true;
      }
    }
    return false;
  }

}
