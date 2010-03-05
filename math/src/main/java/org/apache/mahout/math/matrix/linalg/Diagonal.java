/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.linalg;

import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** For diagonal matrices we can often do better. */
class Diagonal {

  private Diagonal() {
  }

  /**
   * Modifies A to hold its inverse.
   *
   * @return isNonSingular.
   * @throws IllegalArgumentException if <tt>x.size() != y.size()</tt>.
   */
  public static boolean inverse(DoubleMatrix2D A) {
    Property.checkSquare(A);
    boolean isNonSingular = true;
    for (int i = A.rows(); --i >= 0;) {
      double v = A.getQuick(i, i);
      isNonSingular &= (v != 0);
      A.setQuick(i, i, 1 / v);
    }
    return isNonSingular;
  }
}
