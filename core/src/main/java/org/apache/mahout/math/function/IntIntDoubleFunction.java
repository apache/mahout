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
 * Interface that represents a function object: a function that takes three arguments.
 */
public interface IntIntDoubleFunction {

  /**
   * Applies a function to three arguments.
   *
   * @param first  first argument passed to the function.
   * @param second second argument passed to the function.
   * @param third  third argument passed to the function.
   * @return the result of the function.
   */
  double apply(int first, int second, double third);
}
