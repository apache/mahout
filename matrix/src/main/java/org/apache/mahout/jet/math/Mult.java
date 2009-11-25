/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.jet.math;

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
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public final class Mult implements org.apache.mahout.matrix.function.DoubleFunction {
  /**
   * Public read/write access to avoid frequent object construction.
   */
  public double multiplicator;
/**
 * Insert the method's description here.
 * Creation date: (8/10/99 19:12:09)
 */
protected Mult(final double multiplicator) {
  this.multiplicator = multiplicator;
}
/**
 * Returns the result of the function evaluation.
 */
public final double apply(double a) {
  return a * multiplicator;
}
/**
 * <tt>a / constant</tt>.
 */
public static Mult div(final double constant) {
  return mult(1/constant);
}
/**
 * <tt>a * constant</tt>.
 */
public static Mult mult(final double constant) {
  return new Mult(constant);
}
}
