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
 * StudentT distribution (aka T-distribution); See the <A HREF="http://www.cern.ch/RD11/rkb/AN16pp/node279.html#SECTION0002790000000000000000"> math definition</A>
 * and <A HREF="http://www.statsoft.com/textbook/gloss.html#Student's t Distribution"> animated definition</A>.
 * <p>
 * <tt>p(x) = k  *  (1+x^2/f) ^ -(f+1)/2</tt> where <tt>k = g((f+1)/2) / (sqrt(pi*f) * g(f/2))</tt> and <tt>g(a)</tt> being the gamma function and <tt>f</tt> being the degrees of freedom.
 * <p>
 * Valid parameter ranges: <tt>freedom &gt; 0</tt>.
 * <p>
 * Instance methods operate on a user supplied uniform random number generator; they are unsynchronized.
 * <dt>
 * Static methods operate on a default uniform random number generator; they are synchronized.
 * <p>
 * <b>Implementation:</b>
 * <dt>
 * Method: Adapted Polar Box-Muller transformation.
 * <dt>
 * This is a port of <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep/manual/RefGuide/Random/RandStudentT.html">RandStudentT</A> used in <A HREF="http://wwwinfo.cern.ch/asd/lhc++/clhep">CLHEP 1.4.0</A> (C++).
 * CLHEP's implementation, in turn, is based on <tt>tpol.c</tt> from the <A HREF="http://www.cis.tu-graz.ac.at/stat/stadl/random.html">C-RAND / WIN-RAND</A> library.
 * C-RAND's implementation, in turn, is based upon
 * <p>R.W. Bailey (1994): Polar generation of random variates with the t-distribution, Mathematics of Computation 62, 779-781.
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class StudentT extends AbstractContinousDistribution {

  private double freedom;

  private double TERM; // performance cache for pdf()
  // The uniform random number generated shared by all <b>static</b> methods.
  private static final StudentT shared = new StudentT(1.0, makeDefaultGenerator());

  /**
   * Constructs a StudentT distribution. Example: freedom=1.0.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt;= 0.0</tt>.
   */
  public StudentT(double freedom, RandomEngine randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(freedom);
  }

  /** Returns the cumulative distribution function. */
  public double cdf(double x) {
    return Probability.studentT(freedom, x);
  }

  /** Returns a random number from the distribution. */
  @Override
  public double nextDouble() {
    return nextDouble(this.freedom);
  }

  /**
   * Returns a random number from the distribution; bypasses the internal state.
   *
   * @param degreesOfFreedom a degrees of freedom.
   * @throws IllegalArgumentException if <tt>a &lt;= 0.0</tt>.
   */
  public double nextDouble(double degreesOfFreedom) {
    /*
    * The polar method of Box/Muller for generating Normal variates
    * is adapted to the Student-t distribution. The two generated
    * variates are not independent and the expected no. of uniforms
    * per variate is 2.5464.
    *
    * REFERENCE :  - R.W. Bailey (1994): Polar generation of random
    *                variates with the t-distribution, Mathematics
    *                of Computation 62, 779-781.
    */
    if (degreesOfFreedom <= 0.0) {
      throw new IllegalArgumentException();
    }
    double u, v, w;

    do {
      u = 2.0 * randomGenerator.raw() - 1.0;
      v = 2.0 * randomGenerator.raw() - 1.0;
    }
    while ((w = u * u + v * v) > 1.0);

    return (u * Math.sqrt(degreesOfFreedom * (Math.exp(-2.0 / degreesOfFreedom * Math.log(w)) - 1.0) / w));
  }

  /** Returns the probability distribution function. */
  public double pdf(double x) {
    return this.TERM * Math.pow((1 + x * x / freedom), -(freedom + 1) * 0.5);
  }

  /**
   * Sets the distribution parameter.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt;= 0.0</tt>.
   */
  public void setState(double freedom) {
    if (freedom <= 0.0) {
      throw new IllegalArgumentException();
    }
    this.freedom = freedom;

    double val = Fun.logGamma((freedom + 1) / 2) - Fun.logGamma(freedom / 2);
    this.TERM = Math.exp(val) / Math.sqrt(Math.PI * freedom);
  }

  /**
   * Returns a random number from the distribution.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt;= 0.0</tt>.
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

}
