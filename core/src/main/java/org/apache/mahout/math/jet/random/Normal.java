/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.math.jet.stat.Probability;

import java.util.Locale;
import java.util.Random;

/**
 * Implements a normal distribution specified mean and standard deviation.
 */
public class Normal extends AbstractContinousDistribution {

  private double mean;
  private double variance;
  private double standardDeviation;

  private double cache; // cache for Box-Mueller algorithm
  private boolean cacheFilled; // Box-Mueller

  private double normalizer; // performance cache

  /**
   * @param mean               The mean of the resulting distribution.
   * @param standardDeviation  The standard deviation of the distribution.
   * @param randomGenerator    The random number generator to use.  This can be null if you don't
   * need to generate any numbers.
   */
  public Normal(double mean, double standardDeviation, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(mean, standardDeviation);
  }

  /**
   * Returns the cumulative distribution function.
   */
  @Override
  public double cdf(double x) {
    return Probability.normal(mean, variance, x);
  }

  /** Returns the probability density function. */
  @Override
  public double pdf(double x) {
    double diff = x - mean;
    return normalizer * Math.exp(-(diff * diff) / (2.0 * variance));
  }

  /**
   * Returns a random number from the distribution.
   */
  @Override
  public double nextDouble() {
    // Uses polar Box-Muller transformation.
    if (cacheFilled) {
      cacheFilled = false;
      return cache;
    }

    double x;
    double y;
    double r;
    do {
      x = 2.0 * randomDouble() - 1.0;
      y = 2.0 * randomDouble() - 1.0;
      r = x * x + y * y;
    } while (r >= 1.0);

    double z = Math.sqrt(-2.0 * Math.log(r) / r);
    cache = this.mean + this.standardDeviation * x * z;
    cacheFilled = true;
    return this.mean + this.standardDeviation * y * z;
  }

  /** Sets the uniform random generator internally used. */
  @Override
  public final void setRandomGenerator(Random randomGenerator) {
    super.setRandomGenerator(randomGenerator);
    this.cacheFilled = false;
  }

  /**
   * Sets the mean and variance.
   * @param mean The new value for the mean.
   * @param standardDeviation The new value for the standard deviation.
   */
  public final void setState(double mean, double standardDeviation) {
    if (mean != this.mean || standardDeviation != this.standardDeviation) {
      this.mean = mean;
      this.standardDeviation = standardDeviation;
      this.variance = standardDeviation * standardDeviation;
      this.cacheFilled = false;

      this.normalizer = 1.0 / Math.sqrt(2.0 * Math.PI * variance);
    }
  }

  /** Returns a String representation of the receiver. */
  @Override
  public String toString() {
    return String.format(Locale.ENGLISH, "%s(m=%f, sd=%f)", this.getClass().getName(), mean, standardDeviation);
  }
}
