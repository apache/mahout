package org.apache.mahout.math.function;

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/

/**
 * Interface that represents a function object: a function that takes two arguments and returns a single value.
 **/
public interface DoubleDoubleFunction {

  /**
   * Apply the function to the arguments and return the result
   *
   * @param arg1 a double for the first argument
   * @param arg2 a double for the second argument
   * @return the result of applying the function
   */
  double apply(double arg1, double arg2);
}
