/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.IntIntDoubleFunction;
import org.apache.mahout.math.matrix.impl.AbstractMatrix2D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix1D;
import org.apache.mahout.math.matrix.impl.DenseDoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public abstract class DoubleMatrix2D extends AbstractMatrix2D implements Cloneable {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected DoubleMatrix2D() {
  }

  /**
   * Applies a function to each cell and aggregates the results. Returns a value <tt>v</tt> such that
   * <tt>v==a(size())</tt> where <tt>a(i) == aggr( a(i-1), f(get(row,column)) )</tt> and terminators are <tt>a(1) ==
   * f(get(0,0)), a(0)==Double.NaN</tt>. <p> <b>Example:</b>
   * <pre>
   * org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions;
   * 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * // Sum( x[row,col]*x[row,col] )
   * matrix.aggregate(F.plus,F.square);
   * --> 14
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param aggr an aggregation function taking as first argument the current aggregation and as second argument the
   *             transformed current cell value.
   * @param f    a function transforming the current cell value.
   * @return the aggregated measure.
   * @see org.apache.mahout.math.function.Functions
   */
  public double aggregate(DoubleDoubleFunction aggr,
                          DoubleFunction f) {
    if (size() == 0) {
      return Double.NaN;
    }
    double a = f.apply(getQuick(rows - 1, columns - 1));
    int d = 1; // last cell already done
    for (int row = rows; --row >= 0;) {
      for (int column = columns - d; --column >= 0;) {
        a = aggr.apply(a, f.apply(getQuick(row, column)));
      }
      d = 0;
    }
    return a;
  }

  /**
   * Applies a function to each corresponding cell of two matrices and aggregates the results. Returns a value
   * <tt>v</tt> such that <tt>v==a(size())</tt> where <tt>a(i) == aggr( a(i-1), f(get(row,column),other.get(row,column))
   * )</tt> and terminators are <tt>a(1) == f(get(0,0),other.get(0,0)), a(0)==Double.NaN</tt>. <p> <b>Example:</b>
   * <pre>
   * org.apache.mahout.math.function.Functions F = org.apache.mahout.math.function.Functions.functions;
   * x == 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * y == 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * // Sum( x[row,col] * y[row,col] )
   * x.aggregate(y, F.plus, F.mult);
   * --> 14
   *
   * // Sum( (x[row,col] + y[row,col])^2 )
   * x.aggregate(y, F.plus, F.chain(F.square,F.plus));
   * --> 56
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param aggr an aggregation function taking as first argument the current aggregation and as second argument the
   *             transformed current cell values.
   * @param f    a function transforming the current cell values.
   * @return the aggregated measure.
   * @throws IllegalArgumentException if <tt>columns() != other.columns() || rows() != other.rows()</tt>
   * @see org.apache.mahout.math.function.Functions
   */
  public double aggregate(DoubleMatrix2D other, DoubleDoubleFunction aggr,
                          DoubleDoubleFunction f) {
    checkShape(other);
    if (size() == 0) {
      return Double.NaN;
    }
    double a = f.apply(getQuick(rows - 1, columns - 1), other.getQuick(rows - 1, columns - 1));
    int d = 1; // last cell already done
    for (int row = rows; --row >= 0;) {
      for (int column = columns - d; --column >= 0;) {
        a = aggr.apply(a, f.apply(getQuick(row, column), other.getQuick(row, column)));
      }
      d = 0;
    }
    return a;
  }

  /**
   * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of rows and columns as the receiver. <p> The values
   * are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values the values to be filled into the cells.
   * @throws IllegalArgumentException if <tt>values.length != rows() || for any 0 &lt;= row &lt; rows():
   *                                  values[row].length != columns()</tt>.
   */
  public void assign(double[][] values) {
    if (values.length != rows) {
      throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()=" + rows());
    }
    for (int row = rows; --row >= 0;) {
      double[] currentRow = values[row];
      if (currentRow.length != columns) {
        throw new IllegalArgumentException(
            "Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
      }
      for (int column = columns; --column >= 0;) {
        setQuick(row, column, currentRow[column]);
      }
    }
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   */
  public DoubleMatrix2D assign(double value) {
    int r = rows;
    int c = columns;
    //for (int row=rows; --row >= 0;) {
    //  for (int column=columns; --column >= 0;) {
    for (int row = 0; row < r; row++) {
      for (int column = 0; column < c; column++) {
        setQuick(row, column, value);
      }
    }
    return this;
  }

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col])</tt>. <p> <b>Example:</b>
   * <pre>
   * matrix = 2 x 2 matrix
   * 0.5 1.5
   * 2.5 3.5
   *
   * // change each cell to its sine
   * matrix.assign(Functions.sin);
   * -->
   * 2 x 2 matrix
   * 0.479426  0.997495
   * 0.598472 -0.350783
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param function a function object taking as argument the current cell's value.
   * @see org.apache.mahout.math.function.Functions
   */
  public void assign(DoubleFunction function) {
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        setQuick(row, column, function.apply(getQuick(row, column)));
      }
    }
  }

  /**
   * Replaces all cell values of the receiver with the values of another matrix. Both matrices must have the same number
   * of rows and columns. If both matrices share the same cells (as is the case if they are views derived from the same
   * matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep copy of
   * <tt>other</tt>.
   *
   * @param other the source matrix to copy from (may be identical to the receiver).
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>columns() != other.columns() || rows() != other.rows()</tt>
   */
  public DoubleMatrix2D assign(DoubleMatrix2D other) {
    if (other == this) {
      return this;
    }
    checkShape(other);
    if (haveSharedCells(other)) {
      other = other.copy();
    }

    //for (int row=0; row<rows; row++) {
    //for (int column=0; column<columns; column++) {
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        setQuick(row, column, other.getQuick(row, column));
      }
    }
    return this;
  }

  /**
   * Assigns the result of a function to each cell; <tt>x[row,col] = function(x[row,col],y[row,col])</tt>. <p>
   * <b>Example:</b>
   * <pre>
   * // assign x[row,col] = x[row,col]<sup>y[row,col]</sup>
   * m1 = 2 x 2 matrix
   * 0 1
   * 2 3
   *
   * m2 = 2 x 2 matrix
   * 0 2
   * 4 6
   *
   * m1.assign(m2, org.apache.mahout.math.function.Functions.pow);
   * -->
   * m1 == 2 x 2 matrix
   * 1   1
   * 16 729
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>.
   *
   * @param y        the secondary matrix to operate on.
   * @param function a function object taking as first argument the current cell's value of <tt>this</tt>, and as second
   *                 argument the current cell's value of <tt>y</tt>,
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>columns() != other.columns() || rows() != other.rows()</tt>
   * @see org.apache.mahout.math.function.Functions
   */
  public DoubleMatrix2D assign(DoubleMatrix2D y, DoubleDoubleFunction function) {
    checkShape(y);
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        setQuick(row, column, function.apply(getQuick(row, column), y.getQuick(row, column)));
      }
    }
    return this;
  }

  /** Returns the number of cells having non-zero values; ignores tolerance. */
  public int cardinality() {
    int cardinality = 0;
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        if (getQuick(row, column) != 0) {
          cardinality++;
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
  public DoubleMatrix2D copy() {
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
   * Compares this object against the specified object. The result is {@code true} if and only if the argument is
   * not {@code null} and is at least a {@code DoubleMatrix2D} object that has the same number of columns and
   * rows as the receiver and has exactly the same values at the same coordinates.
   *
   * @param obj the object to compare with.
   * @return {@code true} if the objects are the same; {@code false} otherwise.
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DoubleMatrix2D)) {
      return false;
    }

    return org.apache.mahout.math.matrix.linalg.Property.DEFAULT.equals(this, (DoubleMatrix2D) obj);
  }

  /**
   * Assigns the result of a function to each <i>non-zero</i> cell; <tt>x[row,col] = function(x[row,col])</tt>. Use this
   * method for fast special-purpose iteration. If you want to modify another matrix instead of <tt>this</tt> (i.e. work
   * in read-only mode), simply return the input value unchanged.
   *
   * Parameters to function are as follows: <tt>first==row</tt>, <tt>second==column</tt>, <tt>third==nonZeroValue</tt>.
   *
   * @param function a function object taking as argument the current non-zero cell's row, column and value.
   */
  public void forEachNonZero(IntIntDoubleFunction function) {
    for (int row = rows; --row >= 0;) {
      for (int column = columns; --column >= 0;) {
        double value = getQuick(row, column);
        if (value != 0) {
          double r = function.apply(row, column, value);
          if (r != value) {
            setQuick(row, column, r);
          }
        }
      }
    }
  }

  /**
   * Returns the matrix cell value at coordinate <tt>[row,column]</tt>.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @return the value of the specified cell.
   * @throws IndexOutOfBoundsException if <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
   */
  public double get(int row, int column) {
    if (column < 0 || column >= columns || row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("row:" + row + ", column:" + column);
    }
    return getQuick(row, column);
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
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
  public abstract double getQuick(int row, int column);

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  protected boolean haveSharedCells(DoubleMatrix2D other) {
    if (other == null) {
      return false;
    }
    if (this == other) {
      return true;
    }
    return getContent().haveSharedCellsRaw(other.getContent());
  }

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  protected boolean haveSharedCellsRaw(DoubleMatrix2D other) {
    return false;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the same number of
   * rows and columns. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix
   * must also be of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type
   * <tt>SparseDoubleMatrix2D</tt> the new matrix must also be of type <tt>SparseDoubleMatrix2D</tt>, etc. In general,
   * the new matrix should have internal parametrization as similar as possible.
   *
   * @return a new empty matrix of the same dynamic type.
   */
  public DoubleMatrix2D like() {
    return like(rows, columns);
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
  public abstract DoubleMatrix2D like(int rows, int columns);

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the
   * receiver. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be
   * of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix2D</tt> the new
   * matrix must be of type <tt>SparseDoubleMatrix1D</tt>, etc.
   *
   * @param size the number of cells the matrix shall have.
   * @return a new matrix of the corresponding dynamic type.
   */
  public abstract DoubleMatrix1D like1D(int size);

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseDoubleMatrix2D</tt> the new matrix must be of type
   * <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix2D</tt> the new matrix
   * must be of type <tt>SparseDoubleMatrix1D</tt>, etc.
   *
   * @param size   the number of cells the matrix shall have.
   * @param zero   the index of the first element.
   * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @return a new matrix of the corresponding dynamic type.
   */
  protected abstract DoubleMatrix1D like1D(int size, int zero, int stride);

  /**
   * Sets the matrix cell at coordinate <tt>[row,column]</tt> to the specified value.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   * @param value  the value to be filled into the specified cell.
   * @throws IndexOutOfBoundsException if <tt>column&lt;0 || column&gt;=columns() || row&lt;0 || row&gt;=rows()</tt>
   */
  public void set(int row, int column, double value) {
    if (column < 0 || column >= columns || row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("row:" + row + ", column:" + column);
    }
    setQuick(row, column, value);
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
  public abstract void setQuick(int row, int column, double value);

  /**
   * Constructs and returns a 2-dimensional array containing the cell values. The returned array <tt>values</tt> has the
   * form <tt>values[row][column]</tt> and has the same number of rows and columns as the receiver. <p> The values are
   * copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @return an array filled with the values of the cells.
   */
  public double[][] toArray() {
    double[][] values = new double[rows][columns];
    for (int row = rows; --row >= 0;) {
      double[] currentRow = values[row];
      for (int column = columns; --column >= 0;) {
        currentRow[column] = getQuick(row, column);
      }
    }
    return values;
  }

  /**
   * Constructs and returns a new view equal to the receiver. The view is a shallow clone. Calls {@code clone()}
   * and casts the result. <p> <b>Note that the view is not a deep copy.</b> The returned matrix is backed by this
   * matrix, so changes in the returned matrix are reflected in this matrix, and vice-versa. <p> Use {@link #copy()} to
   * construct an independent deep copy rather than a new view.
   *
   * @return a new view of the receiver.
   */
  protected DoubleMatrix2D view() {
    try {
      return (DoubleMatrix2D) clone();
    } catch (CloneNotSupportedException cnse) {
      throw new IllegalStateException();
    }
  }

  /**
   * Constructs and returns a new <i>slice view</i> representing the rows of the given column. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. To obtain a
   * slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>), then apply this method to the
   * sub-range view. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br>
   * 4, 5, 6 </td> <td>viewColumn(0) ==></td> <td valign="top">Matrix1D of size 2:<br> 1, 4</td> </tr> </table>
   *
   * @param column the column to fix.
   * @return a new slice view.
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= columns()</tt>.
   * @see #viewRow(int)
   */
  public DoubleMatrix1D viewColumn(int column) {
    checkColumn(column);
    int viewSize = this.rows;
    int viewZero = index(0, column);
    int viewStride = this.rowStride;
    return like1D(viewSize, viewZero, viewStride);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the column axis. What used to be column <tt>0</tt> is now
   * column <tt>columns()-1</tt>, ..., what used to be column <tt>columns()-1</tt> is now column <tt>0</tt>. The
   * returned view is backed by this matrix, so changes in the returned view are reflected in this matrix, and
   * vice-versa. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4,
   * 5, 6 </td> <td>columnFlip ==></td> <td valign="top">2 x 3 matrix:<br> 3, 2, 1 <br> 6, 5, 4</td> <td>columnFlip
   * ==></td> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new flip view.
   * @see #viewRowFlip()
   */
  public DoubleMatrix2D viewColumnFlip() {
    return (DoubleMatrix2D) view().vColumnFlip();
  }

  /**
   * Constructs and returns a new <i>dice (transposition) view</i>; Swaps axes; example: 3 x 4 matrix --> 4 x 3 matrix.
   * The view has both dimensions exchanged; what used to be columns become rows, what used to be rows become columns.
   * In other words: <tt>view.get(row,column)==this.get(column,row)</tt>. This is a zero-copy transposition, taking
   * O(1), i.e. constant time. The returned view is backed by this matrix, so changes in the returned view are reflected
   * in this matrix, and vice-versa. Use idioms like <tt>result = viewDice(A).copy()</tt> to generate an independent
   * transposed matrix. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2,
   * 3<br> 4, 5, 6 </td> <td>transpose ==></td> <td valign="top">3 x 2 matrix:<br> 1, 4 <br> 2, 5 <br> 3, 6</td>
   * <td>transpose ==></td> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new dice view.
   */
  public DoubleMatrix2D viewDice() {
    return (DoubleMatrix2D) view().vDice();
  }

  /**
   * Constructs and returns a new <i>sub-range view</i> that is a <tt>height x width</tt> sub matrix starting at
   * <tt>[row,column]</tt>.
   *
   * Operations on the returned view can only be applied to the restricted range. Any attempt to access coordinates not
   * contained in the view will throw an <tt>IndexOutOfBoundsException</tt>. <p> <b>Note that the view is really just a
   * range restriction:</b> The returned matrix is backed by this matrix, so changes in the returned matrix are
   * reflected in this matrix, and vice-versa. <p> The view contains the cells from <tt>[row,column]</tt> to
   * <tt>[row+height-1,column+width-1]</tt>, all inclusive. and has <tt>view.rows() == height; view.columns() ==
   * width;</tt>. A view's legal coordinates are again zero based, as usual. In other words, legal coordinates of the
   * view range from <tt>[0,0]</tt> to <tt>[view.rows()-1==height-1,view.columns()-1==width-1]</tt>. As usual, any
   * attempt to access a cell at a coordinate <tt>column&lt;0 || column&gt;=view.columns() || row&lt;0 ||
   * row&gt;=view.rows()</tt> will throw an <tt>IndexOutOfBoundsException</tt>.
   *
   * @param row    The index of the row-coordinate.
   * @param column The index of the column-coordinate.
   * @param height The height of the box.
   * @param width  The width of the box.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>column<0 || width<0 || column+width>columns() || row<0 || height<0 ||
   *                                   row+height>rows()</tt>
   */
  public DoubleMatrix2D viewPart(int row, int column, int height, int width) {
    return (DoubleMatrix2D) view().vPart(row, column, height, width);
  }

  /**
   * Constructs and returns a new <i>slice view</i> representing the columns of the given row. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. To obtain a
   * slice view on subranges, construct a sub-ranging view (<tt>viewPart(...)</tt>), then apply this method to the
   * sub-range view. <p> <b>Example:</b> <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br>
   * 4, 5, 6 </td> <td>viewRow(0) ==></td> <td valign="top">Matrix1D of size 3:<br> 1, 2, 3</td> </tr> </table>
   *
   * @param row the row to fix.
   * @return a new slice view.
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= rows()</tt>.
   * @see #viewColumn(int)
   */
  public DoubleMatrix1D viewRow(int row) {
    checkRow(row);
    int viewSize = this.columns;
    int viewZero = index(row, 0);
    int viewStride = this.columnStride;
    return like1D(viewSize, viewZero, viewStride);
  }

  /**
   * Constructs and returns a new <i>flip view</i> along the row axis. What used to be row <tt>0</tt> is now row
   * <tt>rows()-1</tt>, ..., what used to be row <tt>rows()-1</tt> is now row <tt>0</tt>. The returned view is backed by
   * this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. <p> <b>Example:</b>
   * <table border="0"> <tr nowrap> <td valign="top">2 x 3 matrix: <br> 1, 2, 3<br> 4, 5, 6 </td> <td>rowFlip ==></td>
   * <td valign="top">2 x 3 matrix:<br> 4, 5, 6 <br> 1, 2, 3</td> <td>rowFlip ==></td> <td valign="top">2 x 3 matrix:
   * <br> 1, 2, 3<br> 4, 5, 6 </td> </tr> </table>
   *
   * @return a new flip view.
   * @see #viewColumnFlip()
   */
  public DoubleMatrix2D viewRowFlip() {
    return (DoubleMatrix2D) view().vRowFlip();
  }

  /**
   * Constructs and returns a new <i>selection view</i> that is a matrix holding the indicated cells. There holds
   * <tt>view.rows() == rowIndexes.length, view.columns() == columnIndexes.length</tt> and <tt>view.get(i,j) ==
   * this.get(rowIndexes[i],columnIndexes[j])</tt>. Indexes can occur multiple times and can be in arbitrary order. <p>
   * <b>Example:</b>
   * <pre>
   * this = 2 x 3 matrix:
   * 1, 2, 3
   * 4, 5, 6
   * rowIndexes     = (0,1)
   * columnIndexes  = (1,0,1,0)
   * -->
   * view = 2 x 4 matrix:
   * 2, 1, 2, 1
   * 5, 4, 5, 4
   * </pre>
   * Note that modifying the index arguments after this call has returned has no effect on the view. The returned view
   * is backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa. <p> To
   * indicate "all" rows or "all columns", simply set the respective parameter
   *
   * @param rowIndexes    The rows of the cells that shall be visible in the new view. To indicate that <i>all</i> rows
   *                      shall be visible, simply set this parameter to <tt>null</tt>.
   * @param columnIndexes The columns of the cells that shall be visible in the new view. To indicate that <i>all</i>
   *                      columns shall be visible, simply set this parameter to <tt>null</tt>.
   * @return the new view.
   * @throws IndexOutOfBoundsException
   *  if <tt>!(0 <= rowIndexes[i] < rows())</tt> for any <tt>i=0..rowIndexes.length()-1</tt>.
   * @throws IndexOutOfBoundsException
   *  if <tt>!(0 <= columnIndexes[i] < columns())</tt> for any <tt>i=0..columnIndexes.length()-1</tt>.
   */
  public DoubleMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
    // check for "all"
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

    checkRowIndexes(rowIndexes);
    checkColumnIndexes(columnIndexes);
    int[] rowOffsets = new int[rowIndexes.length];
    int[] columnOffsets = new int[columnIndexes.length];
    for (int i = rowIndexes.length; --i >= 0;) {
      rowOffsets[i] = rowOffset(rowRank(rowIndexes[i]));
    }
    for (int i = columnIndexes.length; --i >= 0;) {
      columnOffsets[i] = columnOffset(columnRank(columnIndexes[i]));
    }
    return viewSelectionLike(rowOffsets, columnOffsets);
  }

  /**
   * Constructs and returns a new <i>selection view</i> that is a matrix holding all <b>rows</b> matching the given
   * condition. Applies the condition to each row and takes only those row where <tt>condition.apply(viewRow(i))</tt>
   * yields <tt>true</tt>. To match columns, use a dice view. <p> <b>Example:</b> <br>
   * <pre>
   * // extract and view all rows which have a value < threshold in the first column (representing "age")
   * final double threshold = 16;
   * matrix.viewSelection(
   * &nbsp;&nbsp;&nbsp;new DoubleMatrix1DProcedure() {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;public final boolean apply(DoubleMatrix1D m) { return m.get(0) < threshold; }
   * &nbsp;&nbsp;&nbsp;}
   * );
   *
   * // extract and view all rows with RMS < threshold
   * // The RMS (Root-Mean-Square) is a measure of the average "size" of the elements of a data sequence.
   * matrix = 0 1 2 3
   * final double threshold = 0.5;
   * matrix.viewSelection(
   * &nbsp;&nbsp;&nbsp;new DoubleMatrix1DProcedure() {
   * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;public final boolean apply(DoubleMatrix1D m) { return
   * Math.sqrt(m.aggregate(F.plus,F.square) / m.size()) < threshold; }
   * &nbsp;&nbsp;&nbsp;}
   * );
   * </pre>
   * For further examples, see the <a href="package-summary.html#FunctionObjects">package doc</a>. The returned view is
   * backed by this matrix, so changes in the returned view are reflected in this matrix, and vice-versa.
   *
   * @param condition The condition to be matched.
   * @return the new view.
   */
  /*
  public DoubleMatrix2D viewSelection(DoubleMatrix1DProcedure condition) {
    IntArrayList matches = new IntArrayList();
    for (int i = 0; i < rows; i++) {
      if (condition.apply(viewRow(i))) {
        matches.add(i);
      }
    }

    matches.trimToSize();
    return viewSelection(matches.elements(), null); // take all columns
  }
   */

  /**
   * Construct and returns a new selection view.
   *
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  protected abstract DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets);


  /**
   * Linear algebraic matrix-vector multiplication; <tt>z = alpha * A * y + beta*z</tt>. <tt>z[i] = alpha*Sum(A[i,j] *
   * y[j]) + beta*z[i], i=0..A.rows()-1, j=0..y.size()-1</tt>. Where <tt>A == this</tt>. <br> Note: Matrix shape
   * conformance is checked <i>after</i> potential transpositions.
   *
   * @param y the source vector.
   * @param z the vector where results are to be stored. Set this parameter to <tt>null</tt> to indicate that a new
   *          result vector shall be constructed.
   * @return z (for convenience only).
   * @throws IllegalArgumentException if <tt>A.columns() != y.size() || A.rows() > z.size())</tt>.
   */
  public DoubleMatrix1D zMult(DoubleMatrix1D y, DoubleMatrix1D z, double alpha, double beta, boolean transposeA) {
    if (transposeA) {
      return viewDice().zMult(y, z, alpha, beta, false);
    }
    //boolean ignore = (z==null);
    if (z == null) {
      z = new DenseDoubleMatrix1D(this.rows);
    }
    if (columns != y.size() || rows > z.size()) {
      throw new IllegalArgumentException("Incompatible args");
    }

    for (int i = rows; --i >= 0;) {
      double s = 0;
      for (int j = columns; --j >= 0;) {
        s += getQuick(i, j) * y.getQuick(j);
      }
      z.setQuick(i, alpha * s + beta * z.getQuick(i));
    }
    return z;
  }

  /**
   * Linear algebraic matrix-matrix multiplication; <tt>C = alpha * A x B + beta*C</tt>. <tt>C[i,j] = alpha*Sum(A[i,k] *
   * B[k,j]) + beta*C[i,j], k=0..n-1</tt>. <br> Matrix shapes: <tt>A(m x n), B(n x p), C(m x p)</tt>. <br> Note: Matrix
   * shape conformance is checked <i>after</i> potential transpositions.
   *
   * @param B the second source matrix.
   * @param C the matrix where results are to be stored. Set this parameter to <tt>null</tt> to indicate that a new
   *          result matrix shall be constructed.
   * @return C (for convenience only).
   * @throws IllegalArgumentException if <tt>B.rows() != A.columns()</tt>.
   * @throws IllegalArgumentException if <tt>C.rows() != A.rows() || C.columns() != B.columns()</tt>.
   * @throws IllegalArgumentException if <tt>A == C || B == C</tt>.
   */
  public DoubleMatrix2D zMult(DoubleMatrix2D B, DoubleMatrix2D C, double alpha, double beta, boolean transposeA,
                              boolean transposeB) {
    if (transposeA) {
      return viewDice().zMult(B, C, alpha, beta, false, transposeB);
    }
    if (transposeB) {
      return this.zMult(B.viewDice(), C, alpha, beta, transposeA, false);
    }

    int m = rows;
    int n = columns;
    int p = B.columns;

    if (C == null) {
      C = new DenseDoubleMatrix2D(m, p);
    }
    if (B.rows != n) {
      throw new IllegalArgumentException("Matrix2D inner dimensions must agree");
    }
    if (C.rows != m || C.columns != p) {
      throw new IllegalArgumentException("Incompatible result matrix");
    }
    if (this == C || B == C) {
      throw new IllegalArgumentException("Matrices must not be identical");
    }

    for (int j = p; --j >= 0;) {
      for (int i = m; --i >= 0;) {
        double s = 0;
        for (int k = n; --k >= 0;) {
          s += getQuick(i, k) * B.getQuick(k, j);
        }
        C.setQuick(i, j, alpha * s + beta * C.getQuick(i, j));
      }
    }
    return C;
  }

  /**
   * Returns the sum of all cells; <tt>Sum( x[i,j] )</tt>.
   *
   * @return the sum.
   */
  public double zSum() {
    if (size() == 0) {
      return 0;
    }
    return aggregate(Functions.PLUS, Functions.IDENTITY);
  }
}
