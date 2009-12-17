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
public class VonMises extends AbstractContinousDistribution {

  private double my_k;

  // cached vars for method nextDouble(a) (for performance only)
  private double k_set = -1.0;
  private double r;

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final VonMises shared = new VonMises(1.0, makeDefaultGenerator());

  /**
   * Constructs a Von Mises distribution. Example: k=1.0.
   *
   * @throws IllegalArgumentException if <tt>k &lt;= 0.0</tt>.
   */
  public VonMises(double freedom, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(freedom);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(this.my_k);
  }

  /**
   * Returns a random number from the distribution; bypasses the internal state.
   *
   * @throws IllegalArgumentException if <tt>k &lt;= 0.0</tt>.
   */
  public double nextDouble(double k) {
/******************************************************************
 *                                                                *
 *         Von Mises Distribution - Acceptance Rejection          *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :  - mwc samples a random number from the von Mises   *
 *               distribution ( -Pi <= x <= Pi) with parameter    *
 *               k > 0  via  rejection from the wrapped Cauchy    *
 *               distibution.                                     *
 * REFERENCE:  - D.J. Best, N.I. Fisher (1979): Efficient         *
 *               simulation of the von Mises distribution,        *
 *               Appl. Statist. 28, 152-157.                      *
 * SUBPROGRAM: - drand(seed) ... (0,1)-Uniform generator with     *
 *               unsigned long integer *seed.                     *
 *                                                                *
 * Implemented by F. Niederl, August 1992                         *
 ******************************************************************/

    if (k <= 0.0) {
      throw new IllegalArgumentException();
    }

    if (k_set != k) {                                               // SET-UP
      double tau = 1.0 + Math.sqrt(1.0 + 4.0 * k * k);
      double rho = (tau - Math.sqrt(2.0 * tau)) / (2.0 * k);
      r = (1.0 + rho * rho) / (2.0 * rho);
      k_set = k;
    }

    // GENERATOR
    double c;
    double w;
    double v;
    do {
      double u = randomGenerator.raw();
      v = randomGenerator.raw();                                // U(0/1)
      double z = Math.cos(Math.PI * u);
      w = (1.0 + r * z) / (r + z);
      c = k * (r - w);
    } while ((c * (2.0 - c) < v) && (Math.log(c / v) + 1.0 < c));         // Acceptance/Rejection

    return (randomGenerator.raw() > 0.5) ? Math.acos(w) : -Math.acos(w);        // Random sign //
    // 0 <= x <= Pi : -Pi <= x <= 0 //
  }

  /**
   * Sets the distribution parameter.
   *
   * @throws IllegalArgumentException if <tt>k &lt;= 0.0</tt>.
   */
  public void setState(double k) {
    if (k <= 0.0) {
      throw new IllegalArgumentException();
    }
    this.my_k = k;
  }

  /**
   * Returns a random number from the distribution.
   *
   * @throws IllegalArgumentException if <tt>k &lt;= 0.0</tt>.
   */
  public static double staticNextDouble(double freedom) {
    synchronized (shared) {
      return shared.nextDouble(freedom);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + my_k + ')';
  }

}
