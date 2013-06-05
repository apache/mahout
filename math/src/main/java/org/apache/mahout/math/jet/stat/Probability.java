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
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat;

import org.apache.mahout.math.jet.random.Normal;

/** Partially deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
public final class Probability {

  private static final Normal UNIT_NORMAL = new Normal(0, 1, null);

  private Probability() {
  }

  /**
   * Returns the area from zero to <tt>x</tt> under the beta density function.
   * <pre>
   *                          x
   *            -             -
   *           | (a+b)       | |  a-1      b-1
   * P(x)  =  ----------     |   t    (1-t)    dt
   *           -     -     | |
   *          | (a) | (b)   -
   *                         0
   * </pre>
   * This function is identical to the incomplete beta integral function <tt>Gamma.incompleteBeta(a, b, x)</tt>.
   *
   * The complemented function is
   *
   * <tt>1 - P(1-x)  =  Gamma.incompleteBeta( b, a, x )</tt>;
   */
  public static double beta(double a, double b, double x) {
    return Gamma.incompleteBeta(a, b, x);
  }

  /**
   * Returns the integral from zero to <tt>x</tt> of the gamma probability density function.
   * <pre>
   *
   *          alpha     - x
   *       beta        |     alpha-1  -beta t
   * y =  ---------    |    t         e        dt
   *       -           |
   *      | (alpha)   -  0
   * </pre>
   * The incomplete gamma integral is used, according to the relation
   *
   * <tt>y = Gamma.incompleteGamma( alpha, beta*x )</tt>.
   *
   * See http://en.wikipedia.org/wiki/Gamma_distribution#Probability_density_function
   *
   * @param alpha the shape parameter of the gamma distribution.
   * @param beta the rate parameter of the gamma distribution.
   * @param x integration end point.
   */
  public static double gamma(double alpha, double beta, double x) {
    if (x < 0.0) {
      return 0.0;
    }
    return Gamma.incompleteGamma(alpha, beta * x);
  }

  /**
   * Returns the sum of the terms <tt>0</tt> through <tt>k</tt> of the Negative Binomial Distribution.
   * <pre>
   *   k
   *   --  ( n+j-1 )   n      j
   *   >   (       )  p  (1-p)
   *   --  (   j   )
   *  j=0
   * </pre>
   * In a sequence of Bernoulli trials, this is the probability that <tt>k</tt> or fewer failures precede the
   * <tt>n</tt>-th success. <p> The terms are not computed individually; instead the incomplete beta integral is
   * employed, according to the formula <p> <tt>y = negativeBinomial( k, n, p ) = Gamma.incompleteBeta( n, k+1, p
   * )</tt>.
   *
   * All arguments must be positive,
   *
   * @param k end term.
   * @param n the number of trials.
   * @param p the probability of success (must be in <tt>(0.0,1.0)</tt>).
   */
  public static double negativeBinomial(int k, int n, double p) {
    if (p < 0.0 || p > 1.0) {
      throw new IllegalArgumentException();
    }
    if (k < 0) {
      return 0.0;
    }

    return Gamma.incompleteBeta(n, k + 1, p);
  }

  /**
   * Returns the area under the Normal (Gaussian) probability density function, integrated from minus infinity to
   * <tt>x</tt> (assumes mean is zero, variance is one).
   * <pre>
   *                            x
   *                             -
   *                   1        | |          2
   *  normal(x)  = ---------    |    exp( - t /2 ) dt
   *               sqrt(2pi)  | |
   *                           -
   *                          -inf.
   * <p/>
   *             =  ( 1 + erf(z) ) / 2
   *             =  erfc(z) / 2
   * </pre>
   * where <tt>z = x/sqrt(2)</tt>. Computation is via the functions <tt>errorFunction</tt> and
   * <tt>errorFunctionComplement</tt>.
   * <p>
   * Computed using method 26.2.17 from Abramovitz and Stegun (see http://www.math.sfu.ca/~cbm/aands/page_932.htm
   * and http://en.wikipedia.org/wiki/Normal_distribution#Numerical_approximations_of_the_normal_cdf
   */

  public static double normal(double a) {
    if (a < 0) {
      return 1 - normal(-a);
    }
    double b0 = 0.2316419;
    double b1 = 0.319381530;
    double b2 = -0.356563782;
    double b3 = 1.781477937;
    double b4 = -1.821255978;
    double b5 = 1.330274429;
    double t = 1 / (1 + b0 * a);
    return 1 - UNIT_NORMAL.pdf(a) * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
  }

  /**
   * Returns the area under the Normal (Gaussian) probability density function, integrated from minus infinity to
   * <tt>x</tt>.
   * <pre>
   *                            x
   *                             -
   *                   1        | |                 2
   *  normal(x)  = ---------    |    exp( - (t-mean) / 2v ) dt
   *               sqrt(2pi*v)| |
   *                           -
   *                          -inf.
   *
   * </pre>
   * where <tt>v = variance</tt>. Computation is via the functions <tt>errorFunction</tt>.
   *
   * @param mean     the mean of the normal distribution.
   * @param variance the variance of the normal distribution.
   * @param x        the integration limit.
   */
  public static double normal(double mean, double variance, double x) {
    return normal((x - mean) / Math.sqrt(variance));
  }

  /**
   * Returns the sum of the first <tt>k</tt> terms of the Poisson distribution.
   * <pre>
   *   k         j
   *   --   -m  m
   *   >   e    --
   *   --       j!
   *  j=0
   * </pre>
   * The terms are not summed directly; instead the incomplete gamma integral is employed, according to the relation <p>
   * <tt>y = poisson( k, m ) = Gamma.incompleteGammaComplement( k+1, m )</tt>.
   *
   * The arguments must both be positive.
   *
   * @param k    number of terms.
   * @param mean the mean of the poisson distribution.
   */
  public static double poisson(int k, double mean) {
    if (mean < 0) {
      throw new IllegalArgumentException();
    }
    if (k < 0) {
      return 0.0;
    }
    return Gamma.incompleteGammaComplement(k + 1, mean);
  }

}
