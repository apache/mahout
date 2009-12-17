/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.doublealgo;

import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix2DProcedure;
import org.apache.mahout.math.matrix.DoubleMatrix3D;
import org.apache.mahout.math.matrix.DoubleMatrix3DProcedure;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Stencil {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Stencil() {
  }

  /**
   * 27 point stencil operation. Applies a function to a moving <tt>3 x 3 x 3</tt> window.
   *
   * @param A                     the matrix to operate on.
   * @param function              the function to be applied to each window.
   * @param maxIterations         the maximum number of times the stencil shall be applied to the matrix. Should be a
   *                              multiple of 2 because two iterations are always done in one atomic step.
   * @param hasConverged          Convergence condition; will return before maxIterations are done when
   *                              <tt>hasConverged.apply(A)==true</tt>. Set this parameter to <tt>null</tt> to indicate
   *                              that no convergence checks shall be made.
   * @param convergenceIterations the number of iterations to pass between each convergence check. (Since a convergence
   *                              may be expensive, you may want to do it only every 2,4 or 8 iterations.)
   * @return the number of iterations actually executed.
   */
  public static int stencil27(DoubleMatrix3D A, org.apache.mahout.math.function.Double27Function function,
                              int maxIterations, DoubleMatrix3DProcedure hasConverged, int convergenceIterations) {
    DoubleMatrix3D B = A.copy();
    if (convergenceIterations <= 1) {
      convergenceIterations = 2;
    }
    if (convergenceIterations % 2 != 0) {
      convergenceIterations++;
    } // odd -> make it even

    int i = 0;
    while (i < maxIterations) { // do two steps at a time for efficiency
      A.zAssign27Neighbors(B, function);
      B.zAssign27Neighbors(A, function);
      i += 2;
      if (i % convergenceIterations == 0 && hasConverged != null) {
        if (hasConverged.apply(A)) {
          return i;
        }
      }
    }
    return i;
  }

  /**
   * 9 point stencil operation. Applies a function to a moving <tt>3 x 3</tt> window.
   *
   * @param A                     the matrix to operate on.
   * @param function              the function to be applied to each window.
   * @param maxIterations         the maximum number of times the stencil shall be applied to the matrix. Should be a
   *                              multiple of 2 because two iterations are always done in one atomic step.
   * @param hasConverged          Convergence condition; will return before maxIterations are done when
   *                              <tt>hasConverged.apply(A)==true</tt>. Set this parameter to <tt>null</tt> to indicate
   *                              that no convergence checks shall be made.
   * @param convergenceIterations the number of iterations to pass between each convergence check. (Since a convergence
   *                              may be expensive, you may want to do it only every 2,4 or 8 iterations.)
   * @return the number of iterations actually executed.
   */
  public static int stencil9(DoubleMatrix2D A, org.apache.mahout.math.function.Double9Function function,
                             int maxIterations, DoubleMatrix2DProcedure hasConverged, int convergenceIterations) {
    DoubleMatrix2D B = A.copy();
    if (convergenceIterations <= 1) {
      convergenceIterations = 2;
    }
    if (convergenceIterations % 2 != 0) {
      convergenceIterations++;
    } // odd -> make it even

    int i = 0;
    while (i < maxIterations) { // do two steps at a time for efficiency
      A.zAssign8Neighbors(B, function);
      B.zAssign8Neighbors(A, function);
      i += 2;
      if (i % convergenceIterations == 0 && hasConverged != null) {
        if (hasConverged.apply(A)) {
          return i;
        }
      }
    }
    return i;
  }
}
