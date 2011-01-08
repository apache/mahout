/*
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

import org.uncommons.maths.random.SecureRandomSeedGenerator;
import org.uncommons.maths.random.SeedException;
import org.uncommons.maths.random.SeedGenerator;

/**
 * Implements an uncommons math compatible generator that avoids /dev/random's tendency to block
 * due to entropy underflow.
 */
public final class FastRandomSeedGenerator implements SeedGenerator {

  private final SeedGenerator[] generators = {new DevURandomSeedGenerator(), new SecureRandomSeedGenerator()};

  /**
   * Generate a seed value for a random number generator.  Try the /dev/urandom generator
   * first, and then fall back to SecureRandomSeedGenerator to guarantee a result.  On
   * platforms with /dev/random, /dev/urandom should exist and thus be fast and pretty good.
   * On platforms without /dev/random, the fallback strategies should also be pretty fast.
   *
   * @param length The length of the seed to generate (in bytes).
   * @return A byte array containing the seed data.
   * @throws org.uncommons.maths.random.SeedException
   *          If a seed cannot be generated for any reason.
   */
  public byte[] generateSeed(int length) throws SeedException {
    SeedException savedException = null;
    for (SeedGenerator generator : generators) {
      try {
        return generator.generateSeed(length);
      } catch (SeedException e) {
        if (savedException != null) {
          savedException.initCause(e);
        }
        savedException = e;
      }
    }
    if (savedException != null) {
      throw savedException;
    } else {
      throw new IllegalStateException("Couldn't generate seed, but didn't find an exception.  Can't happen.");
    }
  }
}
