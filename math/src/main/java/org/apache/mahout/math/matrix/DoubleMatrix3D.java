/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix;

import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.matrix.doublealgo.Formatter;
import org.apache.mahout.math.matrix.impl.AbstractMatrix3D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public abstract class DoubleMatrix3D extends AbstractMatrix3D {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected DoubleMatrix3D() {
  }

  /**
   * Applies a function to each cell and aggregates the results. Returns a value <tt>v</tt> such that
   * <tt>v==a(size())</tt> where <tt>a(i) == aggr( a(i-1), f(get(slice,row,column)) )</tt> and terminators are <tt>a(1)
   * == f(get(0,0,0)), a(0)==Double.NaN</tt>. <p> <b>Example:</b>
   * <pre>
   * org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions;
   * 2 x 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * 4 5
   * 6 7
   *
   * // Sum( x[slice,row,col]*x[slice,row,col] )
   * matrix.aggregate(F.plus,F.square);
   * --> 140
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param aggr an aggregation function taking as first argument the current aggregation and as second argument the
   *             transformed current cell value.
   * @param f    a function transforming the current cell value.
   * @return the aggregated measure.
   * @see org.apache.mahout.math.function.Functions
   */
  public double aggregate(BinaryFunction aggr,
                          UnaryFunction f) {
    if (size() == 0) {
      return Double.NaN;
    }
    double a = f.apply(getQuick(slices - 1, rows - 1, columns - 1));
    int d = 1; // last cell already done
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns - d; --column >= 0;) {
          a = aggr.apply(a, f.apply(getQuick(slice, row, column)));
        }
        d = 0;
      }
    }
    return a;
  }

  /**
   * Applies a function to each corresponding cell of two matrices and aggregates the results. Returns a value
   * <tt>v</tt> such that <tt>v==a(size())</tt> where <tt>a(i) == aggr( a(i-1), f(get(slice,row,column),other.get(slice,row,column))
   * )</tt> and terminators are <tt>a(1) == f(get(0,0,0),other.get(0,0,0)), a(0)==Double.NaN</tt>. <p> <b>Example:</b>
   * <pre>
   * org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions;
   * x = 2 x 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * 4 5
   * 6 7
   *
   * y = 2 x 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * 4 5
   * 6 7
   *
   * // Sum( x[slice,row,col] * y[slice,row,col] )
   * x.aggregate(y, F.plus, F.mult);
   * --> 140
   *
   * // Sum( (x[slice,row,col] + y[slice,row,col])^2 )
   * x.aggregate(y, F.plus, F.chain(F.square,F.plus));
   * --> 560
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param aggr an aggregation function taking as first argument the current aggregation and as second argument the
   *             transformed current cell values.
   * @param f    a function transforming the current cell values.
   * @return the aggregated measure.
   * @throws IllegalArgumentException if <tt>slices() != other.slices() || rows() != other.rows() || columns() !=
   *                                  other.columns()</tt>
   * @see org.apache.mahout.math.function.Functions
   */
  public double aggregate(DoubleMatrix3D other, BinaryFunction aggr,
                          BinaryFunction f) {
    checkShape(other);
    if (size() == 0) {
      return Double.NaN;
    }
    double a = f.apply(getQuick(slices - 1, rows - 1, columns - 1), other.getQuick(slices - 1, rows - 1, columns - 1));
    int d = 1; // last cell already done
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns - d; --column >= 0;) {
          a = aggr.apply(a, f.apply(getQuick(slice, row, column), other.getQuick(slice, row, column)));
        }
        d = 0;
      }
    }
    return a;
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
  public DoubleMatrix3D assign(double[][][] values) {
    if (values.length != slices) {
      throw new IllegalArgumentException(
          "Must have same number of slices: slices=" + values.length + "slices()=" + slices());
    }
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
              "Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
        }
        for (int column = columns; --column >= 0;) {
          setQuick(slice, row, column, currentRow[column]);
        }
      }
    }
    return this;
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   */
  public DoubleMatrix3D assign(double value) {
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          setQuick(slice, row, column, value);
        }
      }
    }
    return this;
  }

  /**
   * Assigns the result of a function to each cell; <tt>x[slice,row,col] = function(x[slice,row,col])</tt>. <p>
   * <b>Example:</b>
   * <pre>
   * matrix = 1 x 2 x 2 matrix
   * 0.5 1.5
   * 2.5 3.5
   *
   * // change each cell to its sine
   * matrix.assign(Functions.sin);
   * -->
   * 1 x 2 x 2 matrix
   * 0.479426  0.997495
   * 0.598472 -0.350783
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param function a function object taking as argument the current cell's value.
   * @return <tt>this</tt> (for convenience only).
   * @see org.apache.mahout.math.function.Functions
   */
  public DoubleMatrix3D assign(UnaryFunction function) {
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          setQuick(slice, row, column, function.apply(getQuick(slice, row, column)));
        }
      }
    }
    return this;
  }

  /**
   * Replaces all cell values of the receiver with the values of another matrix. Both matrices must have the same number
   * of slices, rows and columns. If both matrices share the same cells (as is the case if they are views derived from
   * the same matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep
   * copy of <tt>other</tt>.
   *
   * @param other the source matrix to copy from (may be identical to the receiver).
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>slices() != other.slices() || rows() != other.rows() || columns() !=
   *                                  other.columns()</tt>
   */
  public DoubleMatrix3D assign(DoubleMatrix3D other) {
    if (other == this) {
      return this;
    }
    checkShape(other);
    if (haveSharedCells(other)) {
      other = other.copy();
    }

    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          setQuick(slice, row, column, other.getQuick(slice, row, column));
        }
      }
    }
    return this;
  }

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col],y[row,col])</tt>. <p>
   * <b>Example:</b>
   * <pre>
   * // assign x[row,col] = x[row,col]<sup>y[row,col]</sup>
   * m1 = 1 x 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * m2 = 1 x 2 x 2 matrix
   * 0 2
   * 4 6
   *
   * m1.assign(m2, org.apache.mahout.math.function.Functions.pow);
   * -->
   * m1 == 1 x 2 x 2 matrix
   * 1   1
   * 16 729
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param y        the secondary matrix to operate on.
   * @param function a function object taking as first argument the current cell's value of <tt>this</tt>, and as second
   *                 argument the current cell's value of <tt>y</tt>,
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>slices() != other.slices() || rows() != other.rows() || columns() !=
   *                                  other.columns()</tt>
   * @see org.apache.mahout.math.function.Functions
   */
  public DoubleMatrix3D assign(DoubleMatrix3D y, BinaryFunction function) {
    checkShape(y);
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          setQuick(slice, row, column, function.apply(getQuick(slice, row, column), y.getQuick(slice, row, column)));
        }
      }
    }
    return this;
  }

  /** Returns the number of cells having non-zero values; ignores tolerance. */
  public int cardinality() {
    int cardinality = 0;
    for (int slice = slices; --slice >= 0;) {
      for (int row = rows; --row >= 0;) {
        for (int column = columns; --column >= 0;) {
          if (getQuick(slice, row, column) != 0) {
            cardinality++;
          }
        }
      }
    }
    return cardinality;
  }

  /**
   * Constructs and returns a deep copy of the receiver. <p> <b>Note that the returned matrix is an independent deep
   * copy.</b> The returned matrix is not backed by this matrix, so changes in the returned matrix are not reflected in
   * this matrix, and vice-versa.
   *
   * @return a deep copy of the receiver.
   */
  public DoubleMatrix3D copy() {
    return like().assign(this);
  }

  /**
   * Returns whether all cells are equal to the given value.
   *
   * @param value the value to test against.
   * @return <tt>true</tt> if all cells are equal to the given value, <tt>false</tt> otherwise.
   */
  public boolean equals(double value) {
    return org.apache.mahout.math.matrix.linalg.Property.DEFAULT.equals(this, value);
  }

  /**
   * Compares this object against the specified object. The result is <code>true</code> if and only if the argument is
   * not <code>null</code> and is at least a <code>DoubleMatrix3D</code> object that has the same number of slices, rows
   * and columns as the receiver and has exactly the same values at the same coordinates.
   *
   * @param obj the object to compare with.
   * @return <code>true</code> if the objects are the same; <code>false</code> otherwise.
   */
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DoubleMatrix3D)) {
      return false;
    }

    return org.apache.mahout.math.matrix.linalg.Property.DEFAULT.equals(this, (DoubleMatrix3D) obj);
  }

  /**
   * Returns the matrix cell value at coordinate <tt>[slice,row,column]</tt>.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @return the value of the specified cell.
   * @throws IndexOutOfBoundsException if <tt>slice&lt;0 || slice&gt;=slices() || row&lt;0 || row&gt;=rows() ||
   *                                   column&lt;0 || column&gt;=column()</tt>.
   */
  public double get(int slice, int row, int column) {
    if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
    }
    return getQuick(slice, row, column);
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
  protected DoubleMatrix3D getContent() {
    return this;
  }

  /**
   * Fills the coordinates and values of cells having non-zero values into the specified lists. Fills into the lists,
   * starting at index 0. After this call returns the specified lists all have a new size, the number of non-zero
   * values. <p> In general, fill order is <i>unspecified</i>. This implementation fill like: <tt>for (slice =
   * 0..slices-1) for (row = 0..rows-1) for (column = 0..colums-1) do ... </tt>. However, subclasses are free to us any
   * other order, even an order that may change over time as cell values are changed. (Of course, result lists indexes
   * are guaranteed to correspond to the same cell). For an example, see {@link DoubleMatrix2D#getNonZeros(IntArrayList,IntArrayList,DoubleArrayList)}.
   *
   * @param sliceList  the list to be filled with slice indexes, can have any size.
   * @param rowList    the list to be filled with row indexes, can have any size.
   * @param columnList the list to be filled with column indexes, can have any size.
   * @param valueList  the list to be filled with values, can have any size.
   */
  public void getNonZeros(IntArrayList sliceList, IntArrayList rowList, IntArrayList columnList,
                          DoubleArrayList valueList) {
    sliceList.clear();
    rowList.clear();
    columnList.clear();
    valueList.clear();
    int s = slices;
    int r = rows;
    int c = columns;
    for (int slice = 0; slice < s; slice++) {
      for (int row = 0; row < r; row++) {
        for (int column = 0; column < c; column++) {
          double value = getQuick(slice, row, column);
          if (value != 0) {
            sliceList.add(slice);
            rowList.add(row);
            columnList.add(column);
            valueList.add(value);
          }
        }
      }
    }
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
  public abstract double getQuick(int slice, int row, int column);

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  protected boolean haveSharedCells(DoubleMatrix3D other) {
    if (other == null) {
      return false;
    }
    if (this == other) {
      return true;
    }
    return getContent().haveSharedCellsRaw(other.getContent());
  }

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  protected boolean haveSharedCellsRaw(DoubleMatrix3D other) {
    return false;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the same number of
   * slices, rows and columns. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix3D</tt> the new
   * matrix must also be of type <tt>DenseDoubleMatrix3D</tt>, if the receiver is an instance of type
   * <tt>SparseDoubleMatrix3D</tt> the new matrix must also be of type <tt>SparseDoubleMatrix3D</tt>, etc. In general,
   * the new matrix should have internal parametrization as similar as possible.
   *
   * @return a new empty matrix of the same dynamic type.
   */
  public DoubleMatrix3D like() {
    return like(slices, rows, columns);
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
  public abstract DoubleMatrix3D like(int slices, int rows, int columns);

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
  protected abstract DoubleMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride,
                                           int columnStride);

  /**
   * Sets the matrix cell at coordinate <tt>[slice,row,column]</tt> to the specified value.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @param value  the value to be filled into the specified cell.
   * @throws IndexOutOfBoundsException if <tt>row&lt;0 || row&gt;=rows() || slice&lt;0 || slice&gt;=slices() ||
   *                                   column&lt;0 || column&gt;=column()</tt>.
   */
  public void set(int slice, int row, int column, double value) {
    if (slice < 0 || slice >= slices || row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException("slice:" + slice + ", row:" + row + ", column:" + column);
    }
    setQuick(slice, row, column, value);
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
  public abstract void setQuick(int slice, int row, int column, double value);

  /**
   * Constructs and returns a 2-dimensional array containing the cell values. The returned array <tt>values</tt> has the
   * form <tt>values[slice][row][column]</tt> and has the same number of slices, rows and columns as the receiver. <p>
   * The values are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @return an array filled with the values of the cells.
   */
  public double[][][] toArray() {
    double[][][] values = new double[slices][rows][columns];
    for (int slice = slices; --slice >= 0;) {
      double[][] currentSlice = values[slice];
      for (int row = rows; --row >= 0;) {
        double[] currentRow = currentSlice[row];
        for (int column = columns; --column >= 0;) {
          currentRow[column] = getQuick(slice, row, column);
        }
      }
    }
    return values;
  }

  /**
   * Returns a string representation using default formatting.
   *
   * @see org.apache.mahout.math.matrix.doublealgo.Formatter
   */
  public String toString() {
    return new Formatter().toString(this);
  }

  /**
   * Constructs and returns a new view equal to the receiver. The view is a shallow clone. Calls <code>clone()</code>
   * and casts the result. <p> <b>Note that the view is not a deep copy.</b> The returned matrix is backed by this
   * matrix, so changes in the returned matrix are reflected in this matrix, and vice-versa. <p> Use {@link #copy()} if
   * you want to construct an independent deep copy rather than a new view.
   *
   * @return a new view of the receiver.
   */
  protected DoubleMatrix3D view() {
    return (DoubleMatrix3D) clone();
  }

  /**
   * Constructs and returns a new 2-dimensional <i>slice view</i> representing the slices and rows of the given column.
   * The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. <p> To obtain a slice view on subranges, construct a sub-ranging view (<tt>view().part(...)</tt>), then
   * apply this method to the sub-range view. To obtain 1-dimensional views, apply this method, then apply another slice
   * view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the intermediate 2-dimensional view. To obtain
   * 1-dimensional views on subranges, apply both steps.
   *
   * @param column the index of the column to fix.
   * @return a new 2-dimensional slice view.
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= columns()</tt>.
   * @see #viewSlice(int)
   * @see #viewRow(int)
   */
  public DoubleMatrix2D viewColumn(int column) {
    checkColumn(column);
    int sliceRows = this.slices;
    int sliceColumns = this.rows;

    //int sliceOffset = index(0,0,column);
    int sliceRowZero = sliceZero;
    int sliceColumnZero = rowZero + _columnOffset(_columnRank(column));

    int sliceRowStride = this.sliceStride;
    int sliceColumnStride = this.rowStride;
    return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the column axis. What used to be column <tt>0</tt> is now
   * column <tt>columns()-1</tt>, ..., what used to be column <tt>columns()-1</tt> is now column <tt>0</tt>. The
   * returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa.
   *
   * @return a new flip view.
   * @see #viewSliceFlip()
   * @see #viewRowFlip()
   */
  public DoubleMatrix3D viewColumnFlip() {
    return (DoubleMatrix3D) (view().vColumnFlip());
  }

  /**
   * Constructs and returns a new <i>dice view</i>; Swaps dimensions (axes); Example: 3 x 4 x 5 matrix --> 4 x 3 x 5
   * matrix. The view has dimensions exchanged; what used to be one axis is now another, in all desired permutations.
   * The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa.
   *
   * @param axis0 the axis that shall become axis 0 (legal values 0..2).
   * @param axis1 the axis that shall become axis 1 (legal values 0..2).
   * @param axis2 the axis that shall become axis 2 (legal values 0..2).
   * @return a new dice view.
   * @throws IllegalArgumentException if some of the parameters are equal or not in range 0..2.
   */
  public DoubleMatrix3D viewDice(int axis0, int axis1, int axis2) {
    return (DoubleMatrix3D) (view().vDice(axis0, axis1, axis2));
  }

  /**
   * Constructs and returns a new <i>sub-range view</i> that is a <tt>depth x height x width</tt> sub matrix starting at
   * <tt>[slice,row,column]</tt>; Equivalent to <tt>view().part(slice,row,column,depth,height,width)</tt>; Provided for
   * convenience only. The returned view is backed by this matrix, so changes in the returned view are reflected in this
   * matrix, and vice-versa.
   *
   * @param slice  The index of the slice-coordinate.
   * @param row    The index of the row-coordinate.
   * @param column The index of the column-coordinate.
   * @param depth  The depth of the box.
   * @param height The height of the box.
   * @param width  The width of the box.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>slice<0 || depth<0 || slice+depth>slices() || row<0 || height<0 ||
   *                                   row+height>rows() || column<0 || width<0 || column+width>columns()</tt>
   */
  public DoubleMatrix3D viewPart(int slice, int row, int column, int depth, int height, int width) {
    return (DoubleMatrix3D) (view().vPart(slice, row, column, depth, height, width));
  }

  /**
   * Constructs and returns a new 2-dimensional <i>slice view</i> representing the slices and columns of the given row.
   * The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. <p> To obtain a slice view on subranges, construct a sub-ranging view (<tt>view().part(...)</tt>), then
   * apply this method to the sub-range view. To obtain 1-dimensional views, apply this method, then apply another slice
   * view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the intermediate 2-dimensional view. To obtain
   * 1-dimensional views on subranges, apply both steps.
   *
   * @param row the index of the row to fix.
   * @return a new 2-dimensional slice view.
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= row()</tt>.
   * @see #viewSlice(int)
   * @see #viewColumn(int)
   */
  public DoubleMatrix2D viewRow(int row) {
    checkRow(row);
    int sliceRows = this.slices;
    int sliceColumns = this.columns;

    //int sliceOffset = index(0,row,0);
    int sliceRowZero = sliceZero;
    int sliceColumnZero = columnZero + _rowOffset(_rowRank(row));

    int sliceRowStride = this.sliceStride;
    int sliceColumnStride = this.columnStride;
    return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the row axis. What used to be row <tt>0</tt> is now row
   * <tt>rows()-1</tt>, ..., what used to be row <tt>rows()-1</tt> is now row <tt>0</tt>. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
   *
   * @return a new flip view.
   * @see #viewSliceFlip()
   * @see #viewColumnFlip()
   */
  public DoubleMatrix3D viewRowFlip() {
    return (DoubleMatrix3D) (view().vRowFlip());
  }

  /**
   * Constructs and returns a new <i>selection view</i> that is a matrix holding the indicated cells. There holds
   * <tt>view.slices() == sliceIndexes.length, view.rows() == rowIndexes.length, view.columns() ==
   * columnIndexes.length</tt> and <tt>view.get(k,i,j) == this.get(sliceIndexes[k],rowIndexes[i],columnIndexes[j])</tt>.
   * Indexes can occur multiple times and can be in arbitrary order. For an example see {@link
   * DoubleMatrix2D#viewSelection(int[],int[])}. <p> Note that modifying the index arguments after this call has
   * returned has no effect on the view. The returned view is backed by this matrix, so changes in the returned view are
   * reflected in this matrix, and vice-versa.
   *
   * @param sliceIndexes  The slices of the cells that shall be visible in the new view. To indicate that <i>all</i>
   *                      slices shall be visible, simply set this parameter to <tt>null</tt>.
   * @param rowIndexes    The rows of the cells that shall be visible in the new view. To indicate that <i>all</i> rows
   *                      shall be visible, simply set this parameter to <tt>null</tt>.
   * @param columnIndexes The columns of the cells that shall be visible in the new view. To indicate that <i>all</i>
   *                      columns shall be visible, simply set this parameter to <tt>null</tt>.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>!(0 <= sliceIndexes[i] < slices())</tt> for any <tt>i=0..sliceIndexes.length()-1</tt>.
   * @throws IndexOutOfBoundsException if <tt>!(0 <= rowIndexes[i] < rows())</tt> for any <tt>i=0..rowIndexes.length()-1</tt>.
   * @throws IndexOutOfBoundsException if <tt>!(0 <= columnIndexes[i] < columns())</tt> for any
   *                                   <tt>i=0..columnIndexes.length()-1</tt>.
   */
  public DoubleMatrix3D viewSelection(int[] sliceIndexes, int[] rowIndexes, int[] columnIndexes) {
    // check for "all"
    if (sliceIndexes == null) {
      sliceIndexes = new int[slices];
      for (int i = slices; --i >= 0;) {
        sliceIndexes[i] = i;
      }
    }
    if (rowIndexes == null) {
      rowIndexes = new int[rows];
      for (int i = rows; --i >= 0;) {
        rowIndexes[i] = i;
      }
    }
    if (columnIndexes == null) {
      columnIndexes = new int[columns];
      for (int i = columns; --i >= 0;) {
        columnIndexes[i] = i;
      }
    }

    checkSliceIndexes(sliceIndexes);
    checkRowIndexes(rowIndexes);
    checkColumnIndexes(columnIndexes);

    int[] sliceOffsets = new int[sliceIndexes.length];
    int[] rowOffsets = new int[rowIndexes.length];
    int[] columnOffsets = new int[columnIndexes.length];

    for (int i = sliceIndexes.length; --i >= 0;) {
      sliceOffsets[i] = _sliceOffset(_sliceRank(sliceIndexes[i]));
    }
    for (int i = rowIndexes.length; --i >= 0;) {
      rowOffsets[i] = _rowOffset(_rowRank(rowIndexes[i]));
    }
    for (int i = columnIndexes.length; --i >= 0;) {
      columnOffsets[i] = _columnOffset(_columnRank(columnIndexes[i]));
    }

    return viewSelectionLike(sliceOffsets, rowOffsets, columnOffsets);
  }

  /**
   * Constructs and returns a new <i>selection view</i> that is a matrix holding all <b>slices</b> matching the given
   * condition. Applies the condition to each slice and takes only those where <tt>condition.apply(viewSlice(i))</tt>
   * yields <tt>true</tt>. To match rows or columns, use a dice view. <p> <b>Example:</b> <br>
   * <pre>
   * // extract and view all slices which have an aggregate sum > 1000
   * matrix.viewSelection(
   * &nbsp;&nbsp;&nbsp;new DoubleMatrix2DProcedure() {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;public final boolean apply(DoubleMatrix2D m) { return m.zSum > 1000; }
   * &nbsp;&nbsp;&nbsp;}
   * );
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
   *
   * @param condition The condition to be matched.
   * @return the new view.
   */
  public DoubleMatrix3D viewSelection(DoubleMatrix2DProcedure condition) {
    IntArrayList matches = new IntArrayList();
    for (int i = 0; i < slices; i++) {
      if (condition.apply(viewSlice(i))) {
        matches.add(i);
      }
    }

    matches.trimToSize();
    return viewSelection(matches.elements(), null, null); // take all rows and columns
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param sliceOffsets  the offsets of the visible elements.
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  protected abstract DoubleMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets);

  /**
   * Constructs and returns a new 2-dimensional <i>slice view</i> representing the rows and columns of the given slice.
   * The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. <p> To obtain a slice view on subranges, construct a sub-ranging view (<tt>view().part(...)</tt>), then
   * apply this method to the sub-range view. To obtain 1-dimensional views, apply this method, then apply another slice
   * view (methods <tt>viewColumn</tt>, <tt>viewRow</tt>) on the intermediate 2-dimensional view. To obtain
   * 1-dimensional views on subranges, apply both steps.
   *
   * @param slice the index of the slice to fix.
   * @return a new 2-dimensional slice view.
   * @throws IndexOutOfBoundsException if <tt>slice < 0 || slice >= slices()</tt>.
   * @see #viewRow(int)
   * @see #viewColumn(int)
   */
  public DoubleMatrix2D viewSlice(int slice) {
    checkSlice(slice);
    int sliceRows = this.rows;
    int sliceColumns = this.columns;

    //int sliceOffset = index(slice,0,0);
    int sliceRowZero = rowZero;
    int sliceColumnZero = columnZero + _sliceOffset(_sliceRank(slice));

    int sliceRowStride = this.rowStride;
    int sliceColumnStride = this.columnStride;
    return like2D(sliceRows, sliceColumns, sliceRowZero, sliceColumnZero, sliceRowStride, sliceColumnStride);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the slice axis. What used to be slice <tt>0</tt> is now slice
   * <tt>slices()-1</tt>, ..., what used to be slice <tt>slices()-1</tt> is now slice <tt>0</tt>. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
   *
   * @return a new flip view.
   * @see #viewRowFlip()
   * @see #viewColumnFlip()
   */
  public DoubleMatrix3D viewSliceFlip() {
    return (DoubleMatrix3D) (view().vSliceFlip());
  }

  /**
   * Sorts the matrix slices into ascending order, according to the <i>natural ordering</i> of the matrix values in the
   * given <tt>[row,column]</tt> position. This sort is guaranteed to be <i>stable</i>. For further information, see
   * {@link org.apache.mahout.math.matrix.doublealgo.Sorting#sort(DoubleMatrix3D,int,int)}. For more advanced sorting
   * functionality, see {@link org.apache.mahout.math.matrix.doublealgo.Sorting}.
   *
   * @return a new sorted vector (matrix) view.
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= rows() || column < 0 || column >= columns()</tt>.
   */
  public DoubleMatrix3D viewSorted(int row, int column) {
    return org.apache.mahout.math.matrix.doublealgo.Sorting.mergeSort.sort(this, row, column);
  }

  /**
   * Constructs and returns a new <i>stride view</i> which is a sub matrix consisting of every i-th cell. More
   * specifically, the view has <tt>this.slices()/sliceStride</tt> slices and <tt>this.rows()/rowStride</tt> rows and
   * <tt>this.columns()/columnStride</tt> columns holding cells <tt>this.get(k*sliceStride,i*rowStride,j*columnStride)</tt>
   * for all <tt>k = 0..slices()/sliceStride - 1, i = 0..rows()/rowStride - 1, j = 0..columns()/columnStride - 1</tt>.
   * The returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa.
   *
   * @param sliceStride  the slice step factor.
   * @param rowStride    the row step factor.
   * @param columnStride the column step factor.
   * @return a new view.
   * @throws IndexOutOfBoundsException if <tt>sliceStride<=0 || rowStride<=0 || columnStride<=0</tt>.
   */
  public DoubleMatrix3D viewStrides(int sliceStride, int rowStride, int columnStride) {
    return (DoubleMatrix3D) (view().vStrides(sliceStride, rowStride, columnStride));
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
  public void zAssign27Neighbors(DoubleMatrix3D B, org.apache.mahout.math.function.Double27Function function) {
    if (function == null) {
      throw new IllegalArgumentException("function must not be null.");
    }
    checkShape(B);
    if (rows < 3 || columns < 3 || slices < 3) {
      return;
    } // nothing to do
    int r = rows - 1;
    int c = columns - 1;

    for (int k = 1; k < slices - 1; k++) {
      for (int i = 1; i < r; i++) {
        double a000 = getQuick(k - 1, i - 1, 0);
        double a001 = getQuick(k - 1, i - 1, 1);
        double a010 = getQuick(k - 1, i, 0);
        double a011 = getQuick(k - 1, i, 1);
        double a020 = getQuick(k - 1, i + 1, 0);
        double a021 = getQuick(k - 1, i + 1, 1);

        double a100 = getQuick(k - 1, i - 1, 0);
        double a101 = getQuick(k, i - 1, 1);
        double a110 = getQuick(k, i, 0);
        double a111 = getQuick(k, i, 1);
        double a120 = getQuick(k, i + 1, 0);
        double a121 = getQuick(k, i + 1, 1);

        double a200 = getQuick(k + 1, i - 1, 0);
        double a201 = getQuick(k + 1, i - 1, 1);
        double a210 = getQuick(k + 1, i, 0);
        double a211 = getQuick(k + 1, i, 1);
        double a220 = getQuick(k + 1, i + 1, 0);
        double a221 = getQuick(k + 1, i + 1, 1);

        for (int j = 1; j < c; j++) {
          // in each step 18 cells can be remembered in registers - they don't need to be reread from slow memory
          // in each step 9 instead of 27 cells need to be read from memory.
          double a002 = getQuick(k - 1, i - 1, j + 1);
          double a012 = getQuick(k - 1, i, j + 1);
          double a022 = getQuick(k - 1, i + 1, j + 1);

          double a102 = getQuick(k, i - 1, j + 1);
          double a112 = getQuick(k, i, j + 1);
          double a122 = getQuick(k, i + 1, j + 1);

          double a202 = getQuick(k + 1, i - 1, j + 1);
          double a212 = getQuick(k + 1, i, j + 1);
          double a222 = getQuick(k + 1, i + 1, j + 1);

          B.setQuick(k, i, j, function.apply(
              a000, a001, a002,
              a010, a011, a012,
              a020, a021, a022,

              a100, a101, a102,
              a110, a111, a112,
              a120, a121, a122,

              a200, a201, a202,
              a210, a211, a212,
              a220, a221, a222));

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
      }
    }
  }

  /**
   * Returns the sum of all cells; <tt>Sum( x[i,j,k] )</tt>.
   *
   * @return the sum.
   */
  public double zSum() {
    if (size() == 0) {
      return 0;
    }
    return aggregate(Functions.plus, Functions.identity);
  }
}
