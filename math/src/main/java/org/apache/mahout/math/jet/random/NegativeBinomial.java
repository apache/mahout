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

import org.apache.mahout.math.jet.random.engine.RandomEngine;
import org.apache.mahout.math.jet.stat.Probability;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class NegativeBinomial extends AbstractDiscreteDistribution {

  private int n;
  private double p;

  private Gamma gamma;
  private Poisson poisson;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final NegativeBinomial shared = new NegativeBinomial(1, 0.5, makeDefaultGenerator());

  /**
   * Constructs a Negative Binomial distribution. Example: n=1, p=0.5.
   *
   * @param n               the number of trials.
   * @param p               the probability of success.
   * @param randomGenerator a uniform random number generator.
   */
  public NegativeBinomial(int n, double p, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setNandP(n, p);
    this.gamma = new Gamma(n, 1, randomGenerator);
    this.poisson = new Poisson(0.0, randomGenerator);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(int k) {
    return Probability.negativeBinomial(k, n, p);
  }

  /**
   * Returns a deep copy of the receiver; the copy will produce identical sequences. After this call has returned, the
   * copy and the receiver have equal but separate state.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    NegativeBinomial copy = (NegativeBinomial) super.clone();
    if (this.poisson != null) {
      copy.poisson = (Poisson) this.poisson.clone();
    }
    copy.poisson.setRandomGenerator(copy.getRandomGenerator());
    if (this.gamma != null) {
      copy.gamma = (Gamma) this.gamma.clone();
    }
    copy.gamma.setRandomGenerator(copy.getRandomGenerator());
    return copy;
  }

  /** Returns a random number from the distribution. */
  @Override
  public int nextInt() {
    return nextInt(n, p);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  public int nextInt(int n, double p) {
/******************************************************************
 *                                                                *
 *        Negative Binomial Distribution - Compound method        *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION:    - nbp  samples a random number from the Negative  *
 *                Binomial distribution with parameters r (no. of *
 *                failures given) and p (probability of success)  *
 *                valid for  r > 0, 0 < p < 1.                    *
 *                If G from Gamma(r) then K  from Poiss(pG/(1-p)) *
 *                is NB(r,p)--distributed.                        *
 * REFERENCE:   - J.H. Ahrens, U. Dieter (1974): Computer methods *
 *                for sampling from gamma, beta, Poisson and      *
 *                binomial distributions, Computing 12, 223--246. *
 * SUBPROGRAMS: - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed                     *
 *              - Gamma(seed,a) ... Gamma generator for a > 0     *
 *                unsigned long *seed, double a                   *
 *              - Poisson(seed,a) ...Poisson generator for a > 0  *
 *                unsigned long *seed, double a.                  *
 *                                                                *
 ******************************************************************/

    double x = p / (1.0 - p);
    //double p1 = p;
    double y = x * this.gamma.nextDouble(n, 1);
    return this.poisson.nextInt(y);
  }

  /** Returns the probability distribution function. */
  public double pdf(int k) {
    if (k > n) {
      throw new IllegalArgumentException();
    }
    return org.apache.mahout.math.jet.math.Arithmetic.binomial(n, k) * Math.pow(p, k) * Math.pow(1.0 - p, n - k);
  }

  /**
   * Sets the parameters number of trials and the probability of success.
   *
   * @param n the number of trials
   * @param p the probability of success.
   */
  public void setNandP(int n, double p) {
    this.n = n;
    this.p = p;
  }

  /**
   * Returns a random number from the distribution with the given parameters n and p.
   *
   * @param n the number of trials
   * @param p the probability of success.
   */
  public static int staticNextInt(int n, double p) {
    synchronized (shared) {
      return shared.nextInt(n, p);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + n + ',' + p + ')';
  }

}
