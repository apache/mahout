/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix3D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DenseDoubleMatrix3D extends DoubleMatrix3D {

  /**
   * The elements of this matrix. elements are stored in slice major, then row major, then column major, in order of
   * significance, i.e. index==slice*sliceStride+ row*rowStride + column*columnStride i.e. {slice0 row0..m}, {slice1
   * row0..m}, ..., {sliceN row0..m} with each row storead as {row0 column0..m}, {row1 column0..m}, ..., {rown
   * column0..m}
   */
  protected final double[] elements;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[slice][row][column]</tt> and have exactly the same number of rows in in every slice and exactly the same
   * number of columns in in every row. <p> The values are copied. So subsequent changes in <tt>values</tt> are not
   * reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= slice &lt; values.length: values[slice].length !=
   *                                  values[slice-1].length</tt>.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values[0].length: values[slice][row].length !=
   *                                  values[slice][row-1].length</tt>.
   */
  public DenseDoubleMatrix3D(double[][][] values) {
    this(values.length, (values.length == 0 ? 0 : values[0].length),
        (values.length == 0 ? 0 : values[0].length == 0 ? 0 : values[0][0].length));
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of slices, rows and columns. All entries are initially <tt>0</tt>.
   *
   * @param slices  the number of slices the matrix shall have.
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  public DenseDoubleMatrix3D(int slices, int rows, int columns) {
    setUp(slices, rows, columns);
    this.elements = new double[slices * rows * columns];
  }

  /**
   * Constructs a view with the given parameters.
   *
   * @param slices       the number of slices the matrix shall have.
   * @param rows         the number of rows the matrix shall have.
   * @param columns      the number of columns the matrix shall have.
   * @param elements     the cells.
   * @param sliceZero    the position of the first element.
   * @param rowZero      the position of the first element.
   * @param columnZero   the position of the first element.
   * @param sliceStride  the number of elements between two slices, i.e. <tt>index(k+1,i,j)-index(k,i,j)</tt>.
   * @param rowStride    the number of elements between two rows, i.e. <tt>index(k,i+1,j)-index(k,i,j)</tt>.
   * @param columnStride the number of elements between two columns, i.e. <tt>index(k,i,j+1)-index(k,i,j)</tt>.
   * @throws IllegalArgumentException if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  protected DenseDoubleMatrix3D(int slices, int rows, int columns, double[] elements, int sliceZero, int rowZero,
                                int columnZero, int sliceStride, int rowStride, int columnStride) {
    setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
    this.elements = elements;
    this.isNoView = false;
  }

  /**
   * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt> is required to have the form
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
  @Override
  public DoubleMatrix3D assign(double[][][] values) {
    if (this.isNoView) {
      if (values.length != slices) {
        throw new IllegalArgumentException(
            "Must have same number of slices: slices=" + values.length + "slices()=" + slices());
      }
      int i = slices * rows * columns - columns;
      for (int slice = slices; --slice >= 0;) {
        double[][] currentSlice = values[slice];
        if (currentSlice.length != rows) {
          throw new IllegalArgumentException(
              "Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
        }
        for (int row = rows; --row >= 0;) {
          double[] currentRow = currentSlice[row];
          if (currentRow.length != columns) {
            throw new IllegalArgumentException(
                "Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" +
                    columns());
          }
          System.arraycopy(currentRow, 0, this.elements, i, columns);
          i -= columns;
        }
      }
    } else {
      super.assign(values);
    }
    return this;
  }

  /**
   * Replaces all cell values of the receiver with the values of another matrix. Both matrices must have the same number
   * of slices, rows and columns. If both matrices share the same cells (as is the case if they are views derived from
   * the same matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep
   * copy of <tt>other</tt>.
   *
   * @param source the source matrix to copy from (may be identical to the receiver).
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>slices() != source.slices() || rows() != source.rows() || columns() !=
   *                                  source.columns()</tt>
   */
  @Override
  public DoubleMatrix3D assign(DoubleMatrix3D source) {
    // overriden for performance only
    if (!(source instanceof DenseDoubleMatrix3D)) {
      return super.assign(source);
    }
    DenseDoubleMatrix3D other = (DenseDoubleMatrix3D) source;
    if (other == this) {
      return this;
    }
    checkShape(other);
    if (haveSharedCells(other)) {
      DoubleMatrix3D c = other.copy();
      if (!(c instanceof DenseDoubleMatrix3D)) { // should not happen
        return super.assign(source);
      }
      other = (DenseDoubleMatrix3D) c;
    }

    if (this.isNoView && other.isNoView) { // quickest
      System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
      return this;
    }
    return super.assign(other);
  }

  /**
   * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
   *
   * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 ||
   * column&gt;=column()</tt>.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @return the value at the specified coordinate.
   */
  @Override
  public double getQuick(int slice, int row, int column) {
    //if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows || column<0 || column>=columns) throw new IndexOutOfBoundsException("slice:"+slice+", row:"+row+", column:"+column);
    //return elements[index(slice,row,column)];
    //manually inlined:
    return elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride];
  }

  /**
   * Returns <tt>true</tt> if both matrices share common cells. More formally, returns <tt>true</tt> if <tt>other !=
   * null</tt> and at least one of the following conditions is met <ul> <li>the receiver is a view of the other matrix
   * <li>the other matrix is a view of the receiver <li><tt>this == other</tt> </ul>
   */
  @Override
  protected boolean haveSharedCellsRaw(DoubleMatrix3D other) {
    if (other instanceof SelectedDenseDoubleMatrix3D) {
      SelectedDenseDoubleMatrix3D otherMatrix = (SelectedDenseDoubleMatrix3D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof DenseDoubleMatrix3D) {
      DenseDoubleMatrix3D otherMatrix = (DenseDoubleMatrix3D) other;
      return this.elements == otherMatrix.elements;
    }
    return false;
  }

  /**
   * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the third-coordinate.
   */
  @Override
  protected int index(int slice, int row, int column) {
    //return _sliceOffset(_sliceRank(slice)) + _rowOffset(_rowRank(row)) + _columnOffset(_columnRank(column));
    //manually inlined:
    return sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified
   * number of slices, rows and columns. For example, if the receiver is an instance of type
   * <tt>DenseDoubleMatrix3D</tt> the new matrix must also be of type <tt>DenseDoubleMatrix3D</tt>, if the receiver is
   * an instance of type <tt>SparseDoubleMatrix3D</tt> the new matrix must also be of type
   * <tt>SparseDoubleMatrix3D</tt>, etc. In general, the new matrix should have internal parametrization as similar as
   * possible.
   *
   * @param slices  the number of slices the matrix shall have.
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public DoubleMatrix3D like(int slices, int rows, int columns) {
    return new DenseDoubleMatrix3D(slices, rows, columns);
  }

  /**
   * Construct and returns a new 2-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseDoubleMatrix3D</tt> the new matrix must also be of type
   * <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix3D</tt> the new matrix
   * must also be of type <tt>SparseDoubleMatrix2D</tt>, etc.
   *
   * @param rows         the number of rows the matrix shall have.
   * @param columns      the number of columns the matrix shall have.
   * @param rowZero      the position of the first element.
   * @param columnZero   the position of the first element.
   * @param rowStride    the number of elements between two rows, i.e. <tt>index(i+1,j)-index(i,j)</tt>.
   * @param columnStride the number of elements between two columns, i.e. <tt>index(i,j+1)-index(i,j)</tt>.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  protected DoubleMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
    return new DenseDoubleMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
  }

  /**
   * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the specified value.
   *
   * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() || column&lt;0 ||
   * column&gt;=column()</tt>.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @param value  the value to be filled into the specified cell.
   */
  @Override
  public void setQuick(int slice, int row, int column, double value) {
    //if (debug) if (slice<0 || slice>=slices || row<0 || row>=rows || column<0 || column>=columns) throw new IndexOutOfBoundsException("slice:"+slice+", row:"+row+", column:"+column);
    //elements[index(slice,row,column)] = value;
    //manually inlined:
    elements[sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride] = value;
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param sliceOffsets  the offsets of the visible elements.
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected DoubleMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
    return new SelectedDenseDoubleMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
  }

  /**
   * 27 neighbor stencil transformation. For efficient finite difference operations. Applies a function to a moving
   * <tt>3 x 3 x 3</tt> window. Does nothing if <tt>rows() < 3 || columns() < 3 || slices() < 3</tt>.
   * <pre>
   * B[k,i,j] = function.apply(
   * &nbsp;&nbsp;&nbsp;A[k-1,i-1,j-1], A[k-1,i-1,j], A[k-1,i-1,j+1],
   * &nbsp;&nbsp;&nbsp;A[k-1,i,  j-1], A[k-1,i,  j], A[k-1,i,  j+1],
   * &nbsp;&nbsp;&nbsp;A[k-1,i+1,j-1], A[k-1,i+1,j], A[k-1,i+1,j+1],
   *
   * &nbsp;&nbsp;&nbsp;A[k  ,i-1,j-1], A[k  ,i-1,j], A[k  ,i-1,j+1],
   * &nbsp;&nbsp;&nbsp;A[k  ,i,  j-1], A[k  ,i,  j], A[k  ,i,  j+1],
   * &nbsp;&nbsp;&nbsp;A[k  ,i+1,j-1], A[k  ,i+1,j], A[k  ,i+1,j+1],
   *
   * &nbsp;&nbsp;&nbsp;A[k+1,i-1,j-1], A[k+1,i-1,j], A[k+1,i-1,j+1],
   * &nbsp;&nbsp;&nbsp;A[k+1,i,  j-1], A[k+1,i,  j], A[k+1,i,  j+1],
   * &nbsp;&nbsp;&nbsp;A[k+1,i+1,j-1], A[k+1,i+1,j], A[k+1,i+1,j+1]
   * &nbsp;&nbsp;&nbsp;)
   *
   * x x x - &nbsp;&nbsp;&nbsp; - x x x &nbsp;&nbsp;&nbsp; - - - -
   * x o x - &nbsp;&nbsp;&nbsp; - x o x &nbsp;&nbsp;&nbsp; - - - -
   * x x x - &nbsp;&nbsp;&nbsp; - x x x ... - x x x
   * - - - - &nbsp;&nbsp;&nbsp; - - - - &nbsp;&nbsp;&nbsp; - x o x
   * - - - - &nbsp;&nbsp;&nbsp; - - - - &nbsp;&nbsp;&nbsp; - x x x
   * </pre>
   * Make sure that cells of <tt>this</tt> and <tt>B</tt> do not overlap. In case of overlapping views, behaviour is
   * unspecified. </pre> <p> <b>Example:</b> <pre> final double alpha = 0.25; final double beta = 0.75;
   *
   * org.apache.mahout.math.function.Double27Function f = new Double27Function() { &nbsp;&nbsp;&nbsp;public final
   * double apply( &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a000, double a001, double a002,
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a010, double a011, double a012, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double
   * a020, double a021, double a022,
   *
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a100, double a101, double a102, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double
   * a110, double a111, double a112, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a120, double a121, double a122,
   *
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a200, double a201, double a202, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double
   * a210, double a211, double a212, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;double a220, double a221, double a222) {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return beta*a111 + alpha*(a000 + ... + a222);
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;} }; A.zAssign27Neighbors(B,f); </pre>
   *
   * @param B        the matrix to hold the results.
   * @param function the function to be applied to the 27 cells.
   * @throws NullPointerException     if <tt>function==null</tt>.
   * @throws IllegalArgumentException if <tt>rows() != B.rows() || columns() != B.columns() || slices() != B.slices()
   *                                  </tt>.
   */
  @Override
  public void zAssign27Neighbors(DoubleMatrix3D B, org.apache.mahout.math.function.Double27Function function) {
    // overridden for performance only
    if (!(B instanceof DenseDoubleMatrix3D)) {
      super.zAssign27Neighbors(B, function);
      return;
    }
    if (function == null) {
      throw new IllegalArgumentException("function must not be null.");
    }
    checkShape(B);
    int r = rows - 1;
    int c = columns - 1;
    if (rows < 3 || columns < 3 || slices < 3) {
      return;
    } // nothing to do

    DenseDoubleMatrix3D BB = (DenseDoubleMatrix3D) B;
    int A_ss = sliceStride;
    int A_rs = rowStride;
    int B_rs = BB.rowStride;
    int A_cs = columnStride;
    int B_cs = BB.columnStride;
    double[] elems = this.elements;
    double[] B_elems = BB.elements;
    if (elems == null || B_elems == null) {
      throw new InternalError();
    }

    for (int k = 1; k < slices - 1; k++) {
      int A_index = index(k, 1, 1);
      int B_index = BB.index(k, 1, 1);

      for (int i = 1; i < r; i++) {
        int A002 = A_index - A_ss - A_rs - A_cs;
        int A012 = A002 + A_rs;
        int A022 = A012 + A_rs;

        int A102 = A002 + A_ss;
        int A112 = A102 + A_rs;
        int A122 = A112 + A_rs;

        int A202 = A102 + A_ss;
        int A212 = A202 + A_rs;
        int A222 = A212 + A_rs;

        double a000 = elems[A002];
        A002 += A_cs;
        double a001 = elems[A002];
        double a010 = elems[A012];
        A012 += A_cs;
        double a011 = elems[A012];
        double a020 = elems[A022];
        A022 += A_cs;
        double a021 = elems[A022];

        double a100 = elems[A102];
        A102 += A_cs;
        double a101 = elems[A102];
        double a110 = elems[A112];
        A112 += A_cs;
        double a111 = elems[A112];
        double a120 = elems[A122];
        A122 += A_cs;
        double a121 = elems[A122];

        double a200 = elems[A202];
        A202 += A_cs;
        double a201 = elems[A202];
        double a210 = elems[A212];
        A212 += A_cs;
        double a211 = elems[A212];
        double a220 = elems[A222];
        A222 += A_cs;
        double a221 = elems[A222];

        int B11 = B_index;
        for (int j = 1; j < c; j++) {
          // in each step 18 cells can be remembered in registers - they don't need to be reread from slow memory
          // in each step 9 instead of 27 cells need to be read from memory.
          double a002 = elems[A002 += A_cs];
          double a012 = elems[A012 += A_cs];
          double a022 = elems[A022 += A_cs];

          double a102 = elems[A102 += A_cs];
          double a112 = elems[A112 += A_cs];
          double a122 = elems[A122 += A_cs];

          double a202 = elems[A202 += A_cs];
          double a212 = elems[A212 += A_cs];
          double a222 = elems[A222 += A_cs];

          B_elems[B11] = function.apply(
              a000, a001, a002,
              a010, a011, a012,
              a020, a021, a022,

              a100, a101, a102,
              a110, a111, a112,
              a120, a121, a122,

              a200, a201, a202,
              a210, a211, a212,
              a220, a221, a222);
          B11 += B_cs;

          // move remembered cells
          a000 = a001;
          a001 = a002;
          a010 = a011;
          a011 = a012;
          a020 = a021;
          a021 = a022;

          a100 = a101;
          a101 = a102;
          a110 = a111;
          a111 = a112;
          a120 = a121;
          a121 = a122;

          a200 = a201;
          a201 = a202;
          a210 = a211;
          a211 = a212;
          a220 = a221;
          a221 = a222;
        }
        A_index += A_rs;
        B_index += B_rs;
      }
    }
  }
}
