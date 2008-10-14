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

import org.uncommons.maths.random.MersenneTwisterRNG;

import java.util.Random;

/**
 * <p>The source of random stuff for the whole project. This lets us make all randomness in
 * the project predictable, if desired, for when we run unit tests, which should be repeatable.</p>
 */
public final class RandomUtils {

  private static final byte[] STANDARD_SEED = {
      (byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE,
      (byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE,
      (byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE,
      (byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE,
  };
  private static boolean testSeed;

  private RandomUtils() {
  }

  public static void useTestSeed() {
    testSeed = true;
  }

  public static Random getRandom() {
    return testSeed ? new MersenneTwisterRNG(STANDARD_SEED) : new MersenneTwisterRNG();
  }

  /**
   * @return what {@link Double#hashCode()} would return for the same value
   */
  public static int hashDouble(double value) {
    // Just copied from Double.hashCode
    long bits = Double.doubleToLongBits(value);
    return (int) (bits ^ (bits >>> 32));
  }

}
