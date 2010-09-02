/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.common.RandomUtils;

import java.util.Random;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class PoissonSlow extends AbstractDiscreteDistribution {

  private double mean;

  // precomputed and cached values (for performance only)
  private double cached_sq;
  private double cached_alxm;
  private double cached_g;

  private static final double MEAN_MAX = Integer.MAX_VALUE;
  // for all means larger than that, we don't try to compute a poisson deviation, but return the mean.
  private static final double SWITCH_MEAN = 12.0; // switch from method A to method B

  private static final double[] cof = { // for method logGamma()
      76.18009172947146, -86.50532032941677,
      24.01409824083091, -1.231739572450155,
      0.1208650973866179e-2, -0.5395239384953e-5};

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final PoissonSlow shared = new PoissonSlow(0.0, RandomUtils.getRandom());

  /** Constructs a poisson distribution. Example: mean=1.0. */
  public PoissonSlow(double mean, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setMean(mean);
  }

  /**
   * Returns the value ln(Gamma(xx) for xx > 0.  Full accuracy is obtained for xx > 1. For 0 < xx < 1. the reflection
   * formula (6.1.4) can be used first. (Adapted from Numerical Recipes in C)
   */
  private static double logGamma(double xx) {
    double x = xx - 1.0;
    double tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    double ser = 1.000000000190015;

    double[] coeff = cof;
    for (int j = 0; j <= 5; j++) {
      x++;
      ser += coeff[j] / x;
    }
    return -tmp + Math.log(2.5066282746310005 * ser);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  private int nextInt(double xm) {
    /*
    * Adapted from "Numerical Recipes in C".
    */
    double g = this.cached_g;

    if (xm == -1.0) {
      return 0;
    } // not defined
    if (xm < SWITCH_MEAN) {
      int poisson = -1;
      double product = 1;
      do {
        poisson++;
        product *= randomGenerator.nextDouble();
      } while (product >= g);
      // bug in CLHEP 1.4.0: was "} while ( product > g );"
      return poisson;
    } else if (xm < MEAN_MAX) {
      double t;
      double em;
      double sq = this.cached_sq;
      double alxm = this.cached_alxm;

      Random rand = this.randomGenerator;
      do {
        double y;
        do {
          y = Math.tan(Math.PI * rand.nextDouble());
          em = sq * y + xm;
        } while (em < 0.0);
        em = (double) (int) (em); // faster than em = Math.floor(em); (em>=0.0)
        t = 0.9 * (1.0 + y * y) * Math.exp(em * alxm - logGamma(em + 1.0) - g);
      } while (rand.nextDouble() > t);
      return (int) em;
    } else { // mean is too large
      return (int) xm;
    }
  }

  /** Returns a random number from the distribution. */
  protected int nextIntSlow() {
    double bound = Math.exp(-mean);
    int count = 0;
    double product;
    for (product = 1.0; product >= bound && product > 0.0; count++) {
      product *= randomGenerator.nextDouble();
    }
    if (product <= 0.0 && bound > 0.0) {
      return (int) Math.round(mean);
    } // detected endless loop due to rounding errors
    return count - 1;
  }

  /** Sets the mean. */
  public void setMean(double mean) {
    if (mean != this.mean) {
      this.mean = mean;
      if (mean == -1.0) {
        return;
      } // not defined
      if (mean < SWITCH_MEAN) {
        this.cached_g = Math.exp(-mean);
      } else {
        this.cached_sq = Math.sqrt(2.0 * mean);
        this.cached_alxm = Math.log(mean);
        this.cached_g = mean * cached_alxm - logGamma(mean + 1.0);
      }
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + mean + ')';
  }

}
