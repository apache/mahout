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
 * Interface that represents a function object: a function that takes a single argument and returns a single value.
 * @see org.apache.mahout.math.map
 */
public abstract class DoubleFunction {

  /**
   * Apply the function to the argument and return the result
   *
   * @param x double for the argument
   * @return the result of applying the function
   */
  public abstract double apply(double x);

  public boolean isDensifying() {
    return Math.abs(apply(0.0)) != 0.0;
  }
}
