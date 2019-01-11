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

/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.math.Arithmetic;
import org.apache.mahout.math.jet.stat.Probability;

import java.util.Random;

/** Mostly deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
public final class NegativeBinomial extends AbstractDiscreteDistribution {

  private final int r;
  private final double p;

  private final Gamma gamma;
  private final Poisson poisson;

  /**
   * Constructs a Negative Binomial distribution which describes the probability of getting
   * a particular number of negative trials (k) before getting a fixed number of positive
   * trials (r) where each positive trial has probability (p) of being successful.
   *
   * @param r               the required number of positive trials.
   * @param p               the probability of success.
   * @param randomGenerator a uniform random number generator.
   */
  public NegativeBinomial(int r, double p, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    this.r = r;
    this.p = p;
    this.gamma = new Gamma(r, 1, randomGenerator);
    this.poisson = new Poisson(0.0, randomGenerator);
  }

  /**
   * Returns the cumulative distribution function.
   */
  public double cdf(int k) {
    return Probability.negativeBinomial(k, r, p);
  }

  /**
   * Returns the probability distribution function.
   */
  public double pdf(int k) {
    return Arithmetic.binomial(k + r - 1, r - 1) * Math.pow(p, r) * Math.pow(1.0 - p, k);
  }

  @Override
  public int nextInt() {
    return nextInt(r, p);
  }

  /**
   * Returns a sample from this distribution.  The value returned will
   * be the number of negative samples required before achieving r
   * positive samples.  Each successive sample is taken independently
   * from a Bernouli process with probability p of success.
   *
   * The algorithm used is taken from J.H. Ahrens, U. Dieter (1974):
   * Computer methods for sampling from gamma, beta, Poisson and
   * binomial distributions, Computing 12, 223--246.
   *
   * This algorithm is essentially the same as described at
   * http://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma.E2.80.93Poisson_mixture
   * except that the notion of positive and negative outcomes is uniformly
   * inverted.  Because the inversion is complete and consistent, this
   * definition is effectively identical to that defined on wikipedia.
   */
  public int nextInt(int r, double p) {
    return this.poisson.nextInt(gamma.nextDouble(r, p / (1.0 - p)));
  }

  /**
   * Returns a String representation of the receiver.
   */
  @Override
  public String toString() {
    return this.getClass().getName() + '(' + r + ',' + p + ')';
  }

}
