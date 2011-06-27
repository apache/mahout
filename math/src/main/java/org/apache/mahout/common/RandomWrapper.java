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

import com.google.common.base.Charsets;
import org.uncommons.maths.random.MersenneTwisterRNG;
import org.uncommons.maths.random.RepeatableRNG;
import org.uncommons.maths.random.SeedException;
import org.uncommons.maths.random.SeedGenerator;

import java.util.Random;

public final class RandomWrapper extends Random {

  private static final byte[] STANDARD_SEED = "Mahout=Hadoop+ML".getBytes(Charsets.US_ASCII);
  private static final SeedGenerator SEED_GENERATOR = new FastRandomSeedGenerator();

  private static boolean testSeed;

  private Random random;
  private final Long fixedSeed;

  RandomWrapper() {
    this.fixedSeed = null;
    random = buildRandom();
  }

  RandomWrapper(long fixedSeed) {
    this.fixedSeed = fixedSeed;
    random = buildRandom();
  }

  static void useTestSeed() {
    testSeed = true;
  }

  private Random buildRandom() {
    if (fixedSeed == null) {
      if (testSeed) {
        return new MersenneTwisterRNG(STANDARD_SEED);
      } else {
        // Force use of standard generator, and disallow use of those based on /dev/random since
        // it causes hangs on Ubuntu
        try {
          return new MersenneTwisterRNG(SEED_GENERATOR);
        } catch (SeedException se) {
          // Can't happen
          throw new IllegalStateException(se);
        }
      }
    } else {
      return new MersenneTwisterRNG(RandomUtils.longSeedtoBytes(fixedSeed));
    }
  }

  public Random getRandom() {
    return random;
  }

  void reset() {
    random = buildRandom();
  }

  public long getSeed() {
    return RandomUtils.seedBytesToLong(((RepeatableRNG) random).getSeed());
  }

  @Override
  public void setSeed(long seed) {
    // Since this will be called by the java.util.Random() constructor before we construct
    // the delegate... and because we don't actually care about the result of this for our
    // purpose:
    random = new MersenneTwisterRNG(RandomUtils.longSeedtoBytes(seed));
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
