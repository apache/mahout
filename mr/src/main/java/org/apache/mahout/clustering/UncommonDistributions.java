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

package org.apache.mahout.clustering;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

public final class UncommonDistributions {

  private static final RandomWrapper RANDOM = RandomUtils.getRandom();
  
  private UncommonDistributions() {}
  
  // =============== start of BSD licensed code. See LICENSE.txt
  /**
   * Returns a double sampled according to this distribution. Uniformly fast for all k > 0. (Reference:
   * Non-Uniform Random Variate Generation, Devroye http://cgm.cs.mcgill.ca/~luc/rnbookindex.html) Uses
   * Cheng's rejection algorithm (GB) for k>=1, rejection from Weibull distribution for 0 < k < 1.
   */
  public static double rGamma(double k, double lambda) {
    boolean accept = false;
    if (k >= 1.0) {
      // Cheng's algorithm
      double b = k - Math.log(4.0);
      double c = k + Math.sqrt(2.0 * k - 1.0);
      double lam = Math.sqrt(2.0 * k - 1.0);
      double cheng = 1.0 + Math.log(4.5);
      double x;
      do {
        double u = RANDOM.nextDouble();
        double v = RANDOM.nextDouble();
        double y = 1.0 / lam * Math.log(v / (1.0 - v));
        x = k * Math.exp(y);
        double z = u * v * v;
        double r = b + c * y - x;
        if (r >= 4.5 * z - cheng || r >= Math.log(z)) {
          accept = true;
        }
      } while (!accept);
      return x / lambda;
    } else {
      // Weibull algorithm
      double c = 1.0 / k;
      double d = (1.0 - k) * Math.pow(k, k / (1.0 - k));
      double x;
      do {
        double u = RANDOM.nextDouble();
        double v = RANDOM.nextDouble();
        double z = -Math.log(u);
        double e = -Math.log(v);
        x = Math.pow(z, c);
        if (z + e >= d + x) {
          accept = true;
        }
      } while (!accept);
      return x / lambda;
    }
  }
  
  // ============= end of BSD licensed code
  
  /**
   * Returns a random sample from a beta distribution with the given shapes
   * 
   * @param shape1
   *          a double representing shape1
   * @param shape2
   *          a double representing shape2
   * @return a Vector of samples
   */
  public static double rBeta(double shape1, double shape2) {
    double gam1 = rGamma(shape1, 1.0);
    double gam2 = rGamma(shape2, 1.0);
    return gam1 / (gam1 + gam2);
    
  }
  
  /**
   * Return a random value from a normal distribution with the given mean and standard deviation
   * 
   * @param mean
   *          a double mean value
   * @param sd
   *          a double standard deviation
   * @return a double sample
   */
  public static double rNorm(double mean, double sd) {
    RealDistribution dist = new NormalDistribution(RANDOM.getRandomGenerator(),
                                                   mean,
                                                   sd,
                                                   NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    return dist.sample();
  }
  
  /**
   * Returns an integer sampled according to this distribution. Takes time proportional to np + 1. (Reference:
   * Non-Uniform Random Variate Generation, Devroye http://cgm.cs.mcgill.ca/~luc/rnbookindex.html) Second
   * time-waiting algorithm.
   */
  public static int rBinomial(int n, double p) {
    if (p >= 1.0) {
      return n; // needed to avoid infinite loops and negative results
    }
    double q = -Math.log1p(-p);
    double sum = 0.0;
    int x = 0;
    while (sum <= q) {
      double u = RANDOM.nextDouble();
      double e = -Math.log(u);
      sum += e / (n - x);
      x++;
    }
    if (x == 0) {
      return 0;
    }
    return x - 1;
  }

}
