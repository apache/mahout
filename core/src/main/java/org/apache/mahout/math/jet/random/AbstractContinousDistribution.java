/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

/**
 * Abstract base class for all continuous distributions.  Continuous distributions have
 * probability density and a cumulative distribution functions.
 *
 */
public abstract class AbstractContinousDistribution extends AbstractDistribution {
  public double cdf(double x) {
    throw new UnsupportedOperationException("Can't compute pdf for " + this.getClass().getName());
  }
  
  public double pdf(double x) {
    throw new UnsupportedOperationException("Can't compute pdf for " + this.getClass().getName());
  }

  /**
   * @return A random number from the distribution; returns <tt>(int) Math.round(nextDouble())</tt>.
   *         Override this method if necessary.
   */
  @Override
  public int nextInt() {
    return (int) Math.round(nextDouble());
  }
}
