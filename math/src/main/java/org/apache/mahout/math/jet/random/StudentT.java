/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.jet.stat.Probability;

import java.util.Random;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class StudentT extends AbstractContinousDistribution {

  // The uniform random number generated shared by all <b>static</b> methods.
  private static final StudentT SHARED = new StudentT(1.0, RandomUtils.getRandom());

  private double freedom;
  private double term; // performance cache for pdf()

  /**
   * Constructs a StudentT distribution. Example: freedom=1.0.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt;= 0.0</tt>.
   */
  public StudentT(double freedom, Random randomGenerator) {
    setRandomGenerator(randomGenerator);
    setState(freedom);
  }

  /** Returns the cumulative distribution function. */
  @Override
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
    double u;
    double v;
    double w;

    do {
      u = 2.0 * randomGenerator.nextDouble() - 1.0;
      v = 2.0 * randomGenerator.nextDouble() - 1.0;
    }
    while ((w = u * u + v * v) > 1.0);

    return u * Math.sqrt(degreesOfFreedom * (Math.exp(-2.0 / degreesOfFreedom * Math.log(w)) - 1.0) / w);
  }

  /** Returns the probability distribution function. */
  @Override
  public double pdf(double x) {
    return this.term * Math.pow((1 + x * x / freedom), -(freedom + 1) * 0.5);
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
    this.term = Math.exp(val) / Math.sqrt(Math.PI * freedom);
  }

  /**
   * Returns a random number from the distribution.
   *
   * @param freedom degrees of freedom.
   * @throws IllegalArgumentException if <tt>freedom &lt;= 0.0</tt>.
   */
  public static double staticNextDouble(double freedom) {
    synchronized (SHARED) {
      return SHARED.nextDouble(freedom);
    }
  }

  /** Returns a String representation of the receiver. */
  public String toString() {
    return this.getClass().getName() + '(' + freedom + ')';
  }

}
