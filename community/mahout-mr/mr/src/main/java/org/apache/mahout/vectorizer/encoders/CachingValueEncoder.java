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

package org.apache.mahout.vectorizer.encoders;

import org.apache.mahout.math.MurmurHash;

/**
 * Provides basic hashing semantics for encoders where the probe locations
 * depend only on the name of the variable.
 */
public abstract class CachingValueEncoder extends FeatureVectorEncoder {
  private int[] cachedProbes;

  protected CachingValueEncoder(String name, int seed) {
    super(name);
    cacheProbeLocations(seed);
  }

  /**
   * Sets the number of locations in the feature vector that a value should be in.
   * This causes the cached probe locations to be recomputed.
   *
   * @param probes Number of locations to increment.
   */
  @Override
  public void setProbes(int probes) {
    super.setProbes(probes);
    cacheProbeLocations(getSeed());
  }

  protected abstract int getSeed();

  private void cacheProbeLocations(int seed) {
    cachedProbes = new int[getProbes()];
    for (int i = 0; i < getProbes(); i++) {
      // note that the modulo operation is deferred
      cachedProbes[i] = (int) MurmurHash.hash64A(bytesForString(getName()), seed + i);
    }
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    int h = cachedProbes[probe] % dataSize;
    if (h < 0) {
      h += dataSize;
    }
    return h;
  }
}
