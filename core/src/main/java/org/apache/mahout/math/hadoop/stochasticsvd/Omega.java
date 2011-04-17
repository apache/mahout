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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.util.Arrays;
import java.util.Random;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

/**
 * simplistic implementation for Omega matrix in Stochastic SVD method
 * 
 */
public class Omega {

  private final long seed;
  // TODO find way to make the test pass while using RandomUtils.getRandom()
  private final Random rnd = new Random();
  private final int kp;

  public Omega(long seed, int k, int p) {
    this.seed = seed;
    kp = k + p;

  }

  public void accumDots(int aIndex, double aElement, double[] yRow) {
    rnd.setSeed(getOmegaRowSeed(aIndex, seed, rnd));
    for (int i = 0; i < kp; i++) {
      yRow[i] += rnd.nextGaussian() * aElement;
    }
  }

  /**
   * compute YRow=ARow*Omega.
   * 
   * @param aRow
   *          row of matrix A (size n)
   * @param yRow
   *          row of matrix Y (result) must be pre-allocated to size of (k+p)
   */
  public void computeYRow(Vector aRow, double[] yRow) {
    assert yRow.length == kp;

    Arrays.fill(yRow, 0);
    if ((aRow instanceof SequentialAccessSparseVector) || (aRow instanceof RandomAccessSparseVector)) {
      int j = 0;
      for (Element el : aRow) {
        accumDots(j, el.get(), yRow);
        j++;
      }
      
    } else {
      int n = aRow.size();
      for (int j = 0; j < n; j++) {
        accumDots(j, aRow.getQuick(j), yRow);
      }
    }

  }

  public long getOmegaRowSeed(int omegaRow, long omegaSeed, Random rnd) {
    rnd.setSeed(omegaSeed);
    long rowSeed = rnd.nextLong();
    rnd.setSeed(rowSeed ^ omegaRow);
    return rowSeed ^ rnd.nextLong();

  }

  public static long murmur64(byte[] val, int offset, int len, long seed) {

    long m = 0xc6a4a7935bd1e995L;
    int r = 47;
    long h = seed ^ (len * m);

    int lt = len >>> 3;
    for (int i = 0; i < lt; i++, offset += 8) {
      long k = 0;
      for (int j = 0; j < 8; j++) {
        k <<= 8;
        k |= val[offset + j] & 0xff;
      }

      k *= m;
      k ^= k >>> r;
      k *= m;

      h ^= k;
      h *= m;
    }

    if (offset < len) {
      long k = 0;
      for (; offset < len; offset++) {
        k <<= 8;
        k |= val[offset] & 0xff;
      }
      h ^= k;
      h *= m;
    }

    h ^= h >>> r;
    h *= m;
    h ^= h >>> r;
    return h;

  }

}
