/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.math;

/**
 * Polynomial functions.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Polynomial extends Constants {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected Polynomial() {
  }

  /**
   * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>, assuming coefficient of N is 1.0. Otherwise same
   * as <tt>polevl()</tt>.
   * <pre>
   *                     2          N
   * y  =  C  + C x + C x  +...+ C x
   *        0    1     2          N
   *
   * where C  = 1 and hence is omitted from the array.
   *        N
   *
   * Coefficients are stored in reverse order:
   *
   * coef[0] = C  , ..., coef[N-1] = C  .
   *            N-1                   0
   *
   * Calling arguments are otherwise the same as polevl().
   * </pre>
   * In the interest of speed, there are no checks for out of bounds arithmetic.
   *
   * @param x    argument to the polynomial.
   * @param coef the coefficients of the polynomial.
   * @param N    the degree of the polynomial.
   */
  public static double p1evl(double x, double[] coef, int N) throws ArithmeticException {

    double ans = x + coef[0];

    for (int i = 1; i < N; i++) {
      ans = ans * x + coef[i];
    }

    return ans;
  }

  /**
   * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
   * <pre>
   *                     2          N
   * y  =  C  + C x + C x  +...+ C x
   *        0    1     2          N
   *
   * Coefficients are stored in reverse order:
   *
   * coef[0] = C  , ..., coef[N] = C  .
   *            N                   0
   * </pre>
   * In the interest of speed, there are no checks for out of bounds arithmetic.
   *
   * @param x    argument to the polynomial.
   * @param coef the coefficients of the polynomial.
   * @param N    the degree of the polynomial.
   */
  public static double polevl(double x, double[] coef, int N) throws ArithmeticException {
    double ans = coef[0];

    for (int i = 1; i <= N; i++) {
      ans = ans * x + coef[i];
    }

    return ans;
  }
}
