/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/

package org.apache.mahout.math.function;

/**
 * Only for performance tuning of compute intensive linear algebraic computations.
 * Constructs functions that return one of
 * <ul>
 * <li><tt>a * constant</tt>
 * <li><tt>a / constant</tt>
 * </ul> 
 * <tt>a</tt> is variable, <tt>constant</tt> is fixed, but for performance reasons publicly accessible.
 * Intended to be passed to <tt>matrix.assign(function)</tt> methods.
 */

public final class Mult extends DoubleFunction {

  private double multiplicator;

  Mult(double multiplicator) {
    this.multiplicator = multiplicator;
  }

  /** Returns the result of the function evaluation. */
  @Override
  public double apply(double a) {
    return a * multiplicator;
  }

  /** <tt>a / constant</tt>. */
  public static Mult div(double constant) {
    return mult(1 / constant);
  }

  /** <tt>a * constant</tt>. */
  public static Mult mult(double constant) {
    return new Mult(constant);
  }

  public double getMultiplicator() {
    return multiplicator;
  }

  public void setMultiplicator(double multiplicator) {
    this.multiplicator = multiplicator;
  }
}
