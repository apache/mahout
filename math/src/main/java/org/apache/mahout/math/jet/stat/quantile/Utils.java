/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.jet.stat.quantile;

/** Holds some utility methods shared by different quantile finding implementations. */
final class Utils {

  private Utils() {
  }

  /** Similar to Math.ceil(value), but adjusts small numerical rounding errors +- epsilon. */
  public static long epsilonCeiling(double value) {
    return (long) Math.ceil(value - 0.0000001);
  }
}
