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

public final class RandomWrapper extends Random {

  private static final long STANDARD_SEED = 0xCAFEDEADBEEFBABEL;

  private final RandomGenerator random;

  RandomWrapper() {
    random = new MersenneTwister();
    random.setSeed(System.currentTimeMillis() + System.identityHashCode(random));
  }

  RandomWrapper(long seed) {
    random = new MersenneTwister(seed);
  }

  @Override
  public void setSeed(long seed) {
    // Since this will be called by the java.util.Random() constructor before we construct
    // the delegate... and because we don't actually care about the result of this for our
    // purpose:
    if (random != null) {
      random.setSeed(seed);
    }
  }

  void resetToTestSeed() {
    setSeed(STANDARD_SEED);
  }

  public RandomGenerator getRandomGenerator() {
    return random;
  }

  @Override
  protected int next(int bits) {
    // Ugh, can't delegate this method -- it's protected
    // Callers can't use it and other methods are delegated, so shouldn't matter
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
