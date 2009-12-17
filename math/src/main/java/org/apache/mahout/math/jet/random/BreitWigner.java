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
public class BreitWigner extends AbstractContinousDistribution {

  private double mean;
  private double gamma;
  private double cut;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final BreitWigner shared = new BreitWigner(1.0, 0.2, 1.0, makeDefaultGenerator());

  /**
   * Constructs a BreitWigner distribution.
   *
   * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
   */
  public BreitWigner(double mean, double gamma, double cut, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(mean, gamma, cut);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(mean, gamma, cut);
  }

  /**
   * Returns a random number from the distribution; bypasses the internal state.
   *
   * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
   */
  public double nextDouble(double mean, double gamma, double cut) {

    if (gamma == 0.0) {
      return mean;
    }
    double displ;
    double rval;
    if (cut == Double.NEGATIVE_INFINITY) { // don't cut
      rval = 2.0 * randomGenerator.raw() - 1.0;
      displ = 0.5 * gamma * Math.tan(rval * (Math.PI / 2.0));
      return mean + displ;
    } else {
      double val = Math.atan(2.0 * cut / gamma);
      rval = 2.0 * randomGenerator.raw() - 1.0;
      displ = 0.5 * gamma * Math.tan(rval * val);

      return mean + displ;
    }
  }

  /**
   * Sets the mean, gamma and cut parameters.
   *
   * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
   */
  public void setState(double mean, double gamma, double cut) {
    this.mean = mean;
    this.gamma = gamma;
    this.cut = cut;
  }

  /**
   * Returns a random number from the distribution.
   *
   * @param cut </tt>cut==Double.NEGATIVE_INFINITY</tt> indicates "don't cut".
   */
  public static double staticNextDouble(double mean, double gamma, double cut) {
    synchronized (shared) {
      return shared.nextDouble(mean, gamma, cut);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + mean + ',' + gamma + ',' + cut + ')';
  }

}
