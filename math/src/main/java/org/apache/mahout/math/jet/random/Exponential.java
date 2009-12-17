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

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Exponential extends AbstractContinousDistribution {

  private double lambda;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final Exponential shared = new Exponential(1.0, makeDefaultGenerator());

  /** Constructs a Negative Exponential distribution. */
  public Exponential(double lambda, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(lambda);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(double x) {
    if (x <= 0.0) {
      return 0.0;
    }
    return 1.0 - Math.exp(-x * lambda);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(lambda);
  }

  /** Returns a random number from the distribution; bypasses the internal state. */
  public double nextDouble(double lambda) {
    return -Math.log(randomGenerator.raw()) / lambda;
  }

  /** Returns the probability distribution function. */
  public double pdf(double x) {
    if (x < 0.0) {
      return 0.0;
    }
    return lambda * Math.exp(-x * lambda);
  }

  /** Sets the mean. */
  public void setState(double lambda) {
    this.lambda = lambda;
  }

  /** Returns a random number from the distribution with the given lambda. */
  public static double staticNextDouble(double lambda) {
    synchronized (shared) {
      return shared.nextDouble(lambda);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + lambda + ')';
  }

}
