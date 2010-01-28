/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/**
 * Sparse row-compressed-modified 2-d matrix holding <tt>double</tt> elements.
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 0.9, 04/14/2000
 */
class RCMDoubleMatrix2D extends WrapperDoubleMatrix2D {
  /*
   * The elements of the matrix.
   */
  private final IntArrayList[] indexes;
  private final DoubleArrayList[] values;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  RCMDoubleMatrix2D(double[][] values) {
    this(values.length, values.length == 0 ? 0 : values[0].length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of rows and columns. All entries are initially <tt>0</tt>.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  RCMDoubleMatrix2D(int rows, int columns) {
    super(null);
    setUp(rows, columns);
    indexes = new IntArrayList[rows];
    values = new DoubleArrayList[rows];
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   */
  @Override
  public DoubleMatrix2D assign(double value) {
    // overriden for performance only
    if (value == 0) {
      for (int row = rows; --row >= 0;) {
        indexes[row] = null;
        values[row] = null;
      }
    } else {
      super.assign(value);
    }
    return this;
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
  @Override
  protected DoubleMatrix2D getContent() {
    return this;
  }

  /**
   * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
   *
   * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @return the value at the specified coordinate.
   */
  @Override
  public double getQuick(int row, int column) {
    int k = -1;
    if (indexes[row] != null) {
      k = indexes[row].binarySearch(column);
    }
    if (k < 0) {
      return 0;
    }
    return values[row].getQuick(k);
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified
   * number of rows and columns. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the
   * new matrix must also be of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type
   * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type <tt>SparseDoubleMatrix2D</tt>, etc. In general,
   * the new matrix should have internal parametrization as similar as possible.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public DoubleMatrix2D like(int rows, int columns) {
    return new RCMDoubleMatrix2D(rows, columns);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the
   * receiver. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be
   * of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix2D</tt> the new
   * matrix must be of type <tt>SparseDoubleMatrix1D</tt>, etc.
   *
   * @param size the number of cells the matrix shall have.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  public DoubleMatrix1D like1D(int size) {
    return new SparseDoubleMatrix1D(size);
  }

  /**
   * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified value.
   *
   * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>0 &lt;= column &lt; columns() && 0 &lt;= row &lt; rows()</tt>.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @param value  the value to be filled into the specified cell.
   */
  @Override
  public void setQuick(int row, int column, double value) {
    int i = row;
    int j = column;

    int k = -1;
    IntArrayList indexList = indexes[i];
    if (indexList != null) {
      k = indexList.binarySearch(j);
    }

    if (k >= 0) { // found
      if (value == 0) {
        DoubleArrayList valueList = values[i];
        indexList.remove(k);
        valueList.remove(k);
        int s = indexList.size();
        if (s > 2 && s * 3 < indexList.elements().length) {
          indexList.setSize(s * 3 / 2);
          indexList.trimToSize();
          indexList.setSize(s);

          valueList.setSize(s * 3 / 2);
          valueList.trimToSize();
          valueList.setSize(s);
        }
      } else {
        values[i].setQuick(k, value);
      }
    } else { // not found
      if (value == 0) {
        return;
      }

      k = -k - 1;

      if (indexList == null) {
        indexes[i] = new IntArrayList(3);
        values[i] = new DoubleArrayList(3);
      }
      indexes[i].beforeInsert(k, j);
      values[i].beforeInsert(k, value);
    }
  }

  /**
   * Linear algebraic matrix-vector multiplication; <tt>z = A * y</tt>. <tt>z[i] = alpha*Sum(A[i,j] * y[j]) + beta*z[i],
   * i=0..A.rows()-1, j=0..y.size()-1</tt>. Where <tt>A == this</tt>.
   *
   * @param y the source vector.
   * @param z the vector where results are to be stored.
   * @throws IllegalArgumentException if <tt>A.columns() != y.size() || A.rows() > z.size())</tt>.
   */
  protected void zMult(DoubleMatrix1D y, DoubleMatrix1D z, org.apache.mahout.math.list.IntArrayList nonZeroIndexes,
                       DoubleMatrix1D[] allRows, double alpha, double beta) {
    if (columns != y.size() || rows > z.size()) {
      throw new IllegalArgumentException(
          "Incompatible args: " + toStringShort() + ", " + y.toStringShort() + ", " + z.toStringShort());
    }

    z.assign(Functions.mult(beta / alpha));
    for (int i = indexes.length; --i >= 0;) {
      if (indexes[i] != null) {
        for (int k = indexes[i].size(); --k >= 0;) {
          int j = indexes[i].getQuick(k);
          double value = values[i].getQuick(k);
          z.setQuick(i, z.getQuick(i) + value * y.getQuick(j));
        }
      }
    }

    z.assign(Functions.mult(alpha));
  }
}
