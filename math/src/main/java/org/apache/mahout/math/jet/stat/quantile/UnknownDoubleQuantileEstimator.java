/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.jet.random.engine.RandomEngine;
import org.apache.mahout.math.jet.random.sampling.WeightedRandomSampler;
import org.apache.mahout.math.list.DoubleArrayList;

import java.util.Comparator;

/**
 * Approximate quantile finding algorithm for unknown <tt>N</tt> requiring only one pass and little main memory;
 * computes quantiles over a sequence of <tt>double</tt> elements. This algorithm requires at most two times the memory
 * of a corresponding approx. quantile finder knowing <tt>N</tt>.
 *
 * <p>Needs as input the following parameters:<p> <dt>1. <tt>quantiles</tt> - the number of quantiles to be computed.
 * <dt>2. <tt>epsilon</tt> - the allowed approximation error on quantiles. The approximation guarantee of this algorithm
 * is explicit.
 *
 * <p>It is also possible to couple the approximation algorithm with random sampling to further reduce memory
 * requirements. With sampling, the approximation guarantees are explicit but probabilistic, i.e. they apply with
 * respect to a (user controlled) confidence parameter "delta".
 *
 * <dt>3. <tt>delta</tt> - the probability allowed that the approximation error fails to be smaller than epsilon. Set
 * <tt>delta</tt> to zero for explicit non probabilistic guarantees.
 *
 * You usually don't instantiate quantile finders by using the constructor. Instead use the factory
 * <tt>QuantileFinderFactor</tt> to do so. It will set up the right parametrization for you.
 *
 * <p>After Gurmeet Singh Manku, Sridhar Rajagopalan and Bruce G. Lindsay, Random Sampling Techniques for Space
 * Efficient Online Computation of Order Statistics of Large Datasets. Accepted for Proc. of the 1999 ACM SIGMOD Int.
 * Conf. on Management of Data, Paper (soon) available <A HREF="http://www-cad.eecs.berkeley.edu/~manku"> here</A>.
 *
 * @see QuantileFinderFactory
 */
class UnknownDoubleQuantileEstimator extends DoubleQuantileEstimator {

  private int currentTreeHeight;
  private final int treeHeightStartingSampling;
  private WeightedRandomSampler sampler;
  private final double precomputeEpsilon;

  /**
   * Constructs an approximate quantile finder with b buffers, each having k elements.
   *
   * @param b                 the number of buffers
   * @param k                 the number of elements per buffer
   * @param h                 the tree height at which sampling shall start.
   * @param precomputeEpsilon the epsilon for which quantiles shall be precomputed; set this value <=0.0 if nothing
   *                          shall be precomputed.
   * @param generator         a uniform random number generator.
   */
  UnknownDoubleQuantileEstimator(int b, int k, int h, double precomputeEpsilon, RandomEngine generator) {
    this.sampler = new WeightedRandomSampler(1, generator);
    setUp(b, k);
    this.treeHeightStartingSampling = h;
    this.precomputeEpsilon = precomputeEpsilon;
    this.clear();
  }

  /** Not yet commented. */
  @Override
  protected DoubleBuffer[] buffersToCollapse() {
    DoubleBuffer[] fullBuffers = bufferSet._getFullOrPartialBuffers();

    sortAscendingByLevel(fullBuffers);

    // if there is only one buffer at the lowest level, then increase its level so that there are at least two at the lowest level.
    int minLevel = fullBuffers[1].level();
    if (fullBuffers[0].level() < minLevel) {
      fullBuffers[0].level(minLevel);
    }

    return bufferSet._getFullOrPartialBuffersWithLevel(minLevel);
  }

  /**
   * Removes all elements from the receiver.  The receiver will be empty after this call returns, and its memory
   * requirements will be close to zero.
   */
  @Override
  public synchronized void clear() {
    super.clear();
    this.currentTreeHeight = 1;
    this.sampler.setWeight(1);
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    UnknownDoubleQuantileEstimator copy = (UnknownDoubleQuantileEstimator) super.clone();
    if (this.sampler != null) {
      copy.sampler = (WeightedRandomSampler) copy.sampler.clone();
    }
    return copy;
  }

  /** Not yet commented. */
  @Override
  protected void newBuffer() {
    currentBufferToFill = bufferSet._getFirstEmptyBuffer();
    if (currentBufferToFill == null) {
      throw new IllegalStateException("Oops, no empty buffer.");
    }

    currentBufferToFill.level(currentTreeHeight - 1);
    currentBufferToFill.weight(sampler.getWeight());
  }

  /** Not yet commented. */
  @Override
  protected void postCollapse(DoubleBuffer[] toCollapse) {
    if (toCollapse.length == bufferSet.b()) { //delta for unknown finder
      currentTreeHeight++;
      if (currentTreeHeight >= treeHeightStartingSampling) {
        sampler.setWeight(sampler.getWeight() * 2);
      }
    }
  }

  /**
   * Computes the specified quantile elements over the values previously added.
   *
   * @param phis the quantiles for which elements are to be computed. Each phi must be in the interval (0.0,1.0].
   *             <tt>phis</tt> must be sorted ascending.
   * @return the approximate quantile elements.
   */
  @Override
  public DoubleArrayList quantileElements(DoubleArrayList phis) {
    if (precomputeEpsilon <= 0.0) {
      return super.quantileElements(phis);
    }

    int quantilesToPrecompute = (int) Utils.epsilonCeiling(1.0 / precomputeEpsilon);

    //select that quantile from the precomputed set that corresponds to a position closest to phi.
    phis = phis.copy();
    double e = precomputeEpsilon;
    for (int index = phis.size(); --index >= 0;) {
      double phi = phis.get(index);
      int i = (int) Math.round(((2.0 * phi / e) - 1.0) / 2.0); // finds closest
      i = Math.min(quantilesToPrecompute - 1, Math.max(0, i));
      double augmentedPhi = (e / 2.0) * (1 + 2 * i);
      phis.set(index, augmentedPhi);
    }

    return super.quantileElements(phis);
  }

  /** Not yet commented. */
  @Override
  protected boolean sampleNextElement() {
    return sampler.sampleNextElement();
  }

  /** To do. This could faster be done without sorting (min and second min). */
  private static void sortAscendingByLevel(DoubleBuffer[] fullBuffers) {
    Sorting.quickSort(fullBuffers, 0, fullBuffers.length - 1,
        new Comparator<DoubleBuffer>() {
          @Override
          public int compare(DoubleBuffer o1, DoubleBuffer o2) {
            int l1 = o1.level();
            int l2 = o2.level();
            return l1 < l2 ? -1 : l1 == l2 ? 0 : 1;
          }
        }
    );
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    StringBuffer buf = new StringBuffer(super.toString());
    buf.setLength(buf.length() - 1);
    return buf + ", h=" + currentTreeHeight + ", hStartSampling=" + treeHeightStartingSampling +
        ", precomputeEpsilon=" + precomputeEpsilon + ')';
  }
}
