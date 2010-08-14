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
public class ExponentialPower extends AbstractContinousDistribution {

  private double tau;

  // cached vars for method nextDouble(tau) (for performance only)
  private double s;
  private double sm1;
  private double tau_set = -1.0;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final ExponentialPower shared = new ExponentialPower(1.0, makeDefaultGenerator());

  /**
   * Constructs an Exponential Power distribution. Example: tau=1.0.
   *
   * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
   */
  public ExponentialPower(double tau, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(tau);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(this.tau);
  }

  /**
   * Returns a random number from the distribution; bypasses the internal state.
   *
   * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
   */
  public double nextDouble(double tau) {

    if (tau != tau_set) { // SET-UP
      s = 1.0 / tau;
      sm1 = 1.0 - s;

      tau_set = tau;
    }

    // GENERATOR
    double x;
    double v;
    double u;
    do {
      u = randomGenerator.raw();                             // U(0/1)
      u = (2.0 * u) - 1.0;                                     // U(-1.0/1.0)
      double u1 = Math.abs(u);
      v = randomGenerator.raw();                             // U(0/1)

      if (u1 <= sm1) { // Uniform hat-function for x <= (1-1/tau)
        x = u1;
      } else { // Exponential hat-function for x > (1-1/tau)
        double y = tau * (1.0 - u1);
        x = sm1 - s * Math.log(y);
        v *= y;
      }
    }

    // Acceptance/Rejection
    while (Math.log(v) > -Math.exp(Math.log(x) * tau));

    // Random sign
    return u < 0.0 ? x : -x;
  }

  /**
   * Sets the distribution parameter.
   *
   * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
   */
  public void setState(double tau) {
    if (tau < 1.0) {
      throw new IllegalArgumentException();
    }
    this.tau = tau;
  }

  /**
   * Returns a random number from the distribution.
   *
   * @throws IllegalArgumentException if <tt>tau &lt; 1.0</tt>.
   */
  public static double staticNextDouble(double tau) {
    synchronized (shared) {
      return shared.nextDouble(tau);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + tau + ')';
  }

}
