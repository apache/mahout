/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix3D;
import org.apache.mahout.math.matrix.impl.SparseDoubleMatrix3D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DoubleFactory3D extends PersistentObject {

  /** A factory producing dense matrices. */
  public static final DoubleFactory3D dense = new DoubleFactory3D();

  /** A factory producing sparse matrices. */
  private static final DoubleFactory3D sparse = new DoubleFactory3D();

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected DoubleFactory3D() {
  }

  /** Constructs a matrix with cells having ascending values. For debugging purposes. */
  public DoubleMatrix3D ascending(int slices, int rows, int columns) {
    return descending(slices, rows, columns)
        .assign(Functions.chain(Functions.NEGATE, Functions.minus(slices * rows * columns)));
  }

  /** Constructs a matrix with cells having descending values. For debugging purposes. */
  public DoubleMatrix3D descending(int slices, int rows, int columns) {
    DoubleMatrix3D matrix = make(slices, rows, columns);
    int v = 0;
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          matrix.setQuick(slice, row, column, v++);
        }
      }
    }
    return matrix;
  }

  /**
   * Constructs a matrix with the given cell values. <tt>values</tt> is required to have the form
   * <tt>values[slice][row][column]</tt> and have exactly the same number of slices, rows and columns as the receiver.
   * <p> The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and
   * vice-versa.
   *
   * @param values the values to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>values.length != slices() || for any 0 &lt;= slice &lt; slices():
   *                                  values[slice].length != rows()</tt>.
   * @throws IllegalArgumentException if <tt>for any 0 &lt;= column &lt; columns(): values[slice][row].length !=
   *                                  columns()</tt>.
   */
  public DoubleMatrix3D make(double[][][] values) {
    if (this == sparse) {
      return new SparseDoubleMatrix3D(values);
    }
    return new DenseDoubleMatrix3D(values);
  }

  /** Constructs a matrix with the given shape, each cell initialized with zero. */
  public DoubleMatrix3D make(int slices, int rows, int columns) {
    if (this == sparse) {
      return new SparseDoubleMatrix3D(slices, rows, columns);
    }
    return new DenseDoubleMatrix3D(slices, rows, columns);
  }

  /** Constructs a matrix with the given shape, each cell initialized with the given value. */
  public DoubleMatrix3D make(int slices, int rows, int columns, double initialValue) {
    return make(slices, rows, columns).assign(initialValue);
  }

  /** Constructs a matrix with uniformly distributed values in <tt>(0,1)</tt> (exclusive). */
  public DoubleMatrix3D random(int slices, int rows, int columns) {
    return make(slices, rows, columns).assign(Functions.random());
  }
}
