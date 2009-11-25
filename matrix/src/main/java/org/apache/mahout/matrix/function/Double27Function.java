package org.apache.mahout.matrix.function;

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/

/**
 * Interface that represents a function object: a function that takes 27 arguments and returns a single value.
 *
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public interface Double27Function {

  /**
   * Applies a function to 27 arguments.
   *
   * @return the result of the function.
   */
  double apply(
      double a000, double a001, double a002,
      double a010, double a011, double a012,
      double a020, double a021, double a022,

      double a100, double a101, double a102,
      double a110, double a111, double a112,
      double a120, double a121, double a122,

      double a200, double a201, double a202,
      double a210, double a211, double a212,
      double a220, double a221, double a222
  );
}
