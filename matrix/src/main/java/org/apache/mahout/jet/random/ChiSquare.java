/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.jet.random;

import org.apache.mahout.jet.random.engine.RandomEngine;
import org.apache.mahout.jet.stat.Probability;
/**
 * ChiSquare distribution; See the <A HREF="http://www.cern.ch/RD11/rkb/AN16pp/node31.html#SECTION000310000000000000000"> math definition</A>
 * and <A HREF="http://www.statsoft.com/textbook/glosc.html#Chi-square Distribution"> animated definition</A>.
 * <dt>A special case of the Gamma distribution.
 * <p>
 * <tt>p(x) = (1/g(f/2)) * (x/2)^(f/2-1) * exp(-x/2)</tt> with <tt>g(a)</tt> being the gamma function and <tt>f</tt> being the degrees of freedom.
 * <p>
 * Valid parameter ranges: <tt>freedom &gt; 0</tt>.
 * <p> 
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 * <b>Implementation:</b> 
 * <dt>
 * Method: Ratio of Uniforms with shift.
 * <dt>
 * High performance implementation. This is a port of <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep/manual/RefGuide/Random/RandChiSquare.html">RandChiSquare</A> used in <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep">CLHEP 1.4.0</A> (C++).
 * CLHEP's implementation, in turn, is based on <tt>chru.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library.
 * C-RAND's implementation, in turn, is based upon
 * <p>J.F. Monahan (1987): An algorithm for generating chi random variables, ACM Trans. Math. Software 13, 168-172.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class ChiSquare extends AbstractContinousDistribution {

  protected double freedom;

  // cached vars for method nextDouble(a) (for performance only)
  private double freedom_in = -1.0;
  private double b;
  private double vm;
  private double vd;

  // The uniform random number generated shared by all <b>static</b> methods.
  protected static ChiSquare shared = new ChiSquare(1.0, makeDefaultGenerator());

  /**
   * Constructs a ChiSquare distribution. Example: freedom=1.0.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt; 1.0</tt>.
   */
  public ChiSquare(double freedom, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(freedom);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(double x) {
    return Probability.chiSquare(freedom, x);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(this.freedom);
  }

  /**
   * Returns a random number from the distribution; bypasses the internal state.
   *
   * @param freedom degrees of freedom. It should hold <tt>freedom &lt; 1.0</tt>.
   */
  public double nextDouble(double freedom) {
/******************************************************************
 *                                                                *
 *        Chi Distribution - Ratio of Uniforms  with shift        *
 *                                                                *
 ******************************************************************
 *                                                                *
 * FUNCTION :   - chru samples a random number from the Chi       *
 *                distribution with parameter  a > 1.             *
 * REFERENCE :  - J.F. Monahan (1987): An algorithm for           *
 *                generating chi random variables, ACM Trans.     *
 *                Math. Software 13, 168-172.                     *
 * SUBPROGRAM : - anEngine  ... pointer to a (0,1)-Uniform        *
 *                engine                                          *
 *                                                                *
 * Implemented by R. Kremer, 1990                                 *
 ******************************************************************/

    double u, v, z, zz, r;

    //if( a < 1 )  return (-1.0); // Check for invalid input value

    if (freedom == 1.0) {
      while (true) {
        u = randomGenerator.raw();
        v = randomGenerator.raw() * 0.857763884960707;
        z = v / u;
        if (z < 0) {
          continue;
        }
        zz = z * z;
        r = 2.5 - zz;
        if (z < 0.0) {
          r += zz * z / (3.0 * z);
        }
        if (u < r * 0.3894003915) {
          return (z * z);
        }
        if (zz > (1.036961043 / u + 1.4)) {
          continue;
        }
        if (2.0 * Math.log(u) < (-zz * 0.5)) {
          return (z * z);
        }
      }
    } else {
      if (freedom != freedom_in) {
        b = Math.sqrt(freedom - 1.0);
        vm = -0.6065306597 * (1.0 - 0.25 / (b * b + 1.0));
        vm = (-b > vm) ? -b : vm;
        double vp = 0.6065306597 * (0.7071067812 + b) / (0.5 + b);
        vd = vp - vm;
        freedom_in = freedom;
      }
      while (true) {
        u = randomGenerator.raw();
        v = randomGenerator.raw() * vd + vm;
        z = v / u;
        if (z < -b) {
          continue;
        }
        zz = z * z;
        r = 2.5 - zz;
        if (z < 0.0) {
          r += zz * z / (3.0 * (z + b));
        }
        if (u < r * 0.3894003915) {
          return ((z + b) * (z + b));
        }
        if (zz > (1.036961043 / u + 1.4)) {
          continue;
        }
        if (2.0 * Math.log(u) < (Math.log(1.0 + z / b) * b * b - zz * 0.5 - z * b)) {
          return ((z + b) * (z + b));
        }
      }
    }
  }

  /** Returns the probability distribution function. */
  public double pdf(double x) {
    if (x <= 0.0) {
      throw new IllegalArgumentException();
    }
    double logGamma = Fun.logGamma(freedom / 2.0);
    return Math.exp((freedom / 2.0 - 1.0) * Math.log(x / 2.0) - x / 2.0 - logGamma) / 2.0;
  }

  /**
   * Sets the distribution parameter.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt; 1.0</tt>.
   */
  public void setState(double freedom) {
    if (freedom < 1.0) {
      throw new IllegalArgumentException();
    }
    this.freedom = freedom;
  }

  /**
   * Returns a random number from the distribution.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt; 1.0</tt>.
   */
  public static double staticNextDouble(double freedom) {
    synchronized (shared) {
      return shared.nextDouble(freedom);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + freedom + ')';
  }

  /**
   * Sets the uniform random number generated shared by all <b>static</b> methods.
   *
   * @param randomGenerator the new uniform random number generator to be shared.
   */
  private static void xstaticSetRandomGenerator(RandomEngine randomGenerator) {
    synchronized (shared) {
      shared.setRandomGenerator(randomGenerator);
    }
  }
}
