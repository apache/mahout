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
public class Normal extends AbstractContinousDistribution {

  private double mean;
  private double variance;
  private double standardDeviation;

  private double cache; // cache for Box-Mueller algorithm
  private boolean cacheFilled; // Box-Mueller

  private double normalizer; // performance cache

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final Normal shared = new Normal(0.0, 1.0, makeDefaultGenerator());

  /** Constructs a normal (gauss) distribution. Example: mean=0.0, standardDeviation=1.0. */
  public Normal(double mean, double standardDeviation, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(mean, standardDeviation);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(double x) {
    return Probability.normal(mean, variance, x);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(this.mean, this.standardDeviation);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  public double nextDouble(double mean, double standardDeviation) {
    // Uses polar Box-Muller transformation.
    if (cacheFilled && this.mean == mean && this.standardDeviation == standardDeviation) {
      cacheFilled = false;
      return cache;
    }

    double x;
    double y;
    double r;
    do {
      x = 2.0 * randomGenerator.raw() - 1.0;
      y = 2.0 * randomGenerator.raw() - 1.0;
      r = x * x + y * y;
    } while (r >= 1.0);

    double z = Math.sqrt(-2.0 * Math.log(r) / r);
    cache = mean + standardDeviation * x * z;
    cacheFilled = true;
    return mean + standardDeviation * y * z;
  }

  /** Returns the probability distribution function. */
  public double pdf(double x) {
    double diff = x - mean;
    return normalizer * Math.exp(-(diff * diff) / (2.0 * variance));
  }

  /** Sets the uniform random generator internally used. */
  @Override
  protected void setRandomGenerator(RandomEngine randomGenerator) {
    super.setRandomGenerator(randomGenerator);
    this.cacheFilled = false;
  }

  /** Sets the mean and variance. */
  public void setState(double mean, double standardDeviation) {
    if (mean != this.mean || standardDeviation != this.standardDeviation) {
      this.mean = mean;
      this.standardDeviation = standardDeviation;
      this.variance = standardDeviation * standardDeviation;
      this.cacheFilled = false;

      this.normalizer = 1.0 / Math.sqrt(2.0 * Math.PI * variance);
    }
  }

  /** Returns a random number from the distribution with the given mean and standard deviation. */
  public static double staticNextDouble(double mean, double standardDeviation) {
    synchronized (shared) {
      return shared.nextDouble(mean, standardDeviation);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + mean + ',' + standardDeviation + ')';
  }

}
