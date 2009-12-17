/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface Matrix2DMatrix2DFunction {

  /**
   * Applies a function to two arguments.
   *
   * @param x the first argument passed to the function.
   * @param y the second argument passed to the function.
   * @return the result of the function.
   */
  double apply(DoubleMatrix2D x, DoubleMatrix2D y);
}
