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
package org.apache.mahout.clustering.minhash;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.vectorizer.encoders.MurmurHash;

import java.util.Random;

public class HashFactory {

  private HashFactory() {
  }

  public enum HashType {
    LINEAR, POLYNOMIAL, MURMUR
  }

  public static HashFunction[] createHashFunctions(HashType type, int numFunctions) {
    HashFunction[] hashFunction = new HashFunction[numFunctions];
    Random seed = new Random(11);
    switch (type) {
      case LINEAR:
        for (int i = 0; i < numFunctions; i++) {
          hashFunction[i] = new LinearHash(seed.nextInt(), seed.nextInt());
        }
        break;
      case POLYNOMIAL:
        for (int i = 0; i < numFunctions; i++) {
          hashFunction[i] = new PolynomialHash(seed.nextInt(), seed.nextInt(), seed.nextInt());
        }
        break;
      case MURMUR:
        for (int i = 0; i < numFunctions; i++) {
          hashFunction[i] = new MurmurHashWrapper(seed.nextInt());
        }
        break;
    }
    return hashFunction;
  }

  static class LinearHash implements HashFunction {
    private final int seedA;
    private final int seedB;

    LinearHash(int seedA, int seedB) {
      this.seedA = seedA;
      this.seedB = seedB;
    }

    @Override
    public int hash(byte[] bytes) {
      long hashValue = 31;
      for (long byteVal : bytes) {
        hashValue *= seedA * byteVal;
        hashValue += seedB;
      }
      return Math.abs((int) (hashValue % RandomUtils.MAX_INT_SMALLER_TWIN_PRIME));
    }
  }

  static class PolynomialHash implements HashFunction {
    private final int seedA;
    private final int seedB;
    private final int seedC;

    PolynomialHash(int seedA, int seedB, int seedC) {
      this.seedA = seedA;
      this.seedB = seedB;
      this.seedC = seedC;
    }

    @Override
    public int hash(byte[] bytes) {
      long hashValue = 31;
      for (long byteVal : bytes) {
        hashValue *= seedA * (byteVal >> 4);
        hashValue += seedB * byteVal + seedC;
      }
      return Math
          .abs((int) (hashValue % RandomUtils.MAX_INT_SMALLER_TWIN_PRIME));
    }
  }

  static class MurmurHashWrapper implements HashFunction {
    private final int seed;

    MurmurHashWrapper(int seed) {
      this.seed = seed;
    }

    @Override
    public int hash(byte[] bytes) {
      long hashValue = MurmurHash.hash64A(bytes, seed);
      return Math.abs((int) (hashValue % RandomUtils.MAX_INT_SMALLER_TWIN_PRIME));
    }
  }
}
