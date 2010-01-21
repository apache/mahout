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
public class Hyperbolic extends AbstractContinousDistribution {

  private double alpha;
  private double beta;

  // cached values shared for generateHyperbolic(...)
  private double a_setup = 0.0;
  private double b_setup = -1.0;
  private double hr;
  private double hl;
  private double s;
  private double pm;
  private double pr;
  private double samb;
  private double pmr;
  private double mpa_1;
  private double mmb_1;


  // The uniform random number generated shared by all <b>static</b> methods.
  private static final Hyperbolic shared = new Hyperbolic(10.0, 10.0, makeDefaultGenerator());

  /** Constructs a Beta distribution. */
  public Hyperbolic(double alpha, double beta, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(alpha, beta);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(alpha, beta);
  }

  /** Returns a hyperbolic distributed random number; bypasses the internal state. */
  public double nextDouble(double alpha, double beta) {
/******************************************************************
 *                                                                *
 *        Hyperbolic Distribution - Non-Universal Rejection       *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION   : - hyplc.c samples a random number from the        *
 *                hyperbolic distribution with shape parameter a  *
 *                and b valid for a>0 and |b|<a using the         *
 *                non-universal rejection method for log-concave  *
 *                densities.                                      *
 * REFERENCE :  - L. Devroye (1986): Non-Uniform Random Variate   *
 *                Generation, Springer Verlag, New York.          *
 * SUBPROGRAM : - drand(seed) ... (0,1)-Uniform generator with    *
 *                unsigned long integer *seed.                    *
 *                                                                *
 ******************************************************************/

    if ((a_setup != alpha) || (b_setup != beta)) { // SET-UP
      //double pl;
      double amb = alpha * alpha - beta * beta;
      samb = Math.sqrt(amb);                                  // -log(f(mode))
      double mode = beta / samb;
      double help_1 = alpha * Math.sqrt(2.0 * samb + 1.0);
      double help_2 = beta * (samb + 1.0);
      double mpa = (help_2 + help_1) / amb;
      double mmb = (help_2 - help_1) / amb;
      double a_ = mpa - mode;
      double b_ = -mmb + mode;
      hr = -1.0 / (-alpha * mpa / Math.sqrt(1.0 + mpa * mpa) + beta);
      hl = 1.0 / (-alpha * mmb / Math.sqrt(1.0 + mmb * mmb) + beta);
      double a_1 = a_ - hr;
      double b_1 = b_ - hl;
      mmb_1 = mode - b_1;                                     // lower border
      mpa_1 = mode + a_1;                                     // upper border

      s = (a_ + b_);
      pm = (a_1 + b_1) / s;
      pr = hr / s;
      pmr = pm + pr;

      a_setup = alpha;
      b_setup = beta;
    }

    // GENERATOR
    double x;
    while (true) {
      double u = randomGenerator.raw();
      double v = randomGenerator.raw();
      if (u <= pm) { // Rejection with a uniform majorizing function
        // over the body of the distribution
        x = mmb_1 + u * s;
        if (Math.log(v) <= (-alpha * Math.sqrt(1.0 + x * x) + beta * x + samb)) {
          break;
        }
      } else {
        double e;
        if (u <= pmr) {  // Rejection with an exponential envelope on the
          // right side of the mode
          e = -Math.log((u - pm) / pr);
          x = mpa_1 + hr * e;
          if ((Math.log(v) - e) <= (-alpha * Math.sqrt(1.0 + x * x) + beta * x + samb)) {
            break;
          }
        } else {           // Rejection with an exponential envelope on the
          // left side of the mode
          e = Math.log((u - pmr) / (1.0 - pmr));
          x = mmb_1 + hl * e;
          if ((Math.log(v) + e) <= (-alpha * Math.sqrt(1.0 + x * x) + beta * x + samb)) {
            break;
          }
        }
      }
    }

    return (x);
  }

  /** Sets the parameters. */
  public void setState(double alpha, double beta) {
    this.alpha = alpha;
    this.beta = beta;
  }

  /** Returns a random number from the distribution. */
  public static double staticNextDouble(double alpha, double beta) {
    synchronized (shared) {
      return shared.nextDouble(alpha, beta);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + alpha + ',' + beta + ')';
  }

}
