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

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.Random;

/**
 * A wrapper around a {@link RandomGenerator} that implements {@link Random} so that it can be
 * used in places that expect a {@link Random} instance.
 */
public final class RandomWrapper extends Random {

  /**
   * A standard seed that can be used to ensure that tests are deterministic.
   */
  private static final long STANDARD_SEED = 0xCAFEDEADBEEFBABEL;

  /**
   * The delegate random number generator.
   */
  private final RandomGenerator random;

  /**
   * Creates a new random number generator.
   */
  RandomWrapper() {
    random = new MersenneTwister();
    random.setSeed(System.currentTimeMillis() + System.identityHashCode(random));
  }

  RandomWrapper(long seed) {
    random = new MersenneTwister(seed);
  }

  /**
   * Creates a new random number generator.
   *
   * @param seed the seed for the random number generator.
   */
  @Override
  public void setSeed(long seed) {
    // Since this will be called by the java.util.Random() constructor before we construct
    // the delegate... and because we don't actually care about the result of this for our
    // purpose:
    if (random != null) {
      random.setSeed(seed);
    }
  }

    /**
     * Sets the seed to a standard value that can be used to ensure that tests are deterministic.
     */
  void resetToTestSeed() {
    setSeed(STANDARD_SEED);
  }

    /**
     * @return the delegate random number generator.
     */
  public RandomGenerator getRandomGenerator() {
    return random;
  }

  /**
   * This method is not supported. It will throw an {@link UnsupportedOperationException}.
   * Reason: This method is protected in {@link Random} and cannot be delegated.
   *
   * Callers can't use this method and other methods are delegated, so shouldn't matter.
   *
   * @param bits - why are you still reading?
   * @return An error, don't use it.
   */
  @Override
  protected int next(int bits) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void nextBytes(byte[] bytes) {
    random.nextBytes(bytes);
  }

  @Override
  public int nextInt() {
    return random.nextInt();
  }

  @Override
  public int nextInt(int n) {
    return random.nextInt(n);
  }

  @Override
  public long nextLong() {
    return random.nextLong();
  }

  @Override
  public boolean nextBoolean() {
    return random.nextBoolean();
  }

  @Override
  public float nextFloat() {
    return random.nextFloat();
  }

  @Override
  public double nextDouble() {
    return random.nextDouble();
  }

  @Override
  public double nextGaussian() {
    return random.nextGaussian();
  }

}
