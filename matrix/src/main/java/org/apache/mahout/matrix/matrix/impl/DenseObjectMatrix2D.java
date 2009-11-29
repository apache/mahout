/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.impl;

import org.apache.mahout.matrix.function.ObjectFunction;
import org.apache.mahout.matrix.matrix.ObjectMatrix1D;
import org.apache.mahout.matrix.matrix.ObjectMatrix2D;
/**
 Dense 2-d matrix holding <tt>Object</tt> elements.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Implementation:</b>
 <p>
 Internally holds one single contigous one-dimensional array, addressed in row major.
 Note that this implementation is not synchronized.
 <p>
 <b>Memory requirements:</b>
 <p>
 <tt>memory [bytes] = 8*rows()*columns()</tt>.
 Thus, a 1000*1000 matrix uses 8 MB.
 <p>
 <b>Time complexity:</b>
 <p>
 <tt>O(1)</tt> (i.e. constant time) for the basic operations
 <tt>get</tt>, <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 <p>
 Cells are internally addressed in row-major.
 Applications demanding utmost speed can exploit this fact.
 Setting/getting values in a loop row-by-row is quicker than column-by-column.
 Thus
 <pre>
 for (int row=0; row < rows; row++) {
 for (int column=0; column < columns; column++) {
 matrix.setQuick(row,column,someValue);
 }
 }
 </pre>
 is quicker than
 <pre>
 for (int column=0; column < columns; column++) {
 for (int row=0; row < rows; row++) {
 matrix.setQuick(row,column,someValue);
 }
 }
 </pre>
 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DenseObjectMatrix2D extends ObjectMatrix2D {

  /**
   * The elements of this matrix. elements are stored in row major, i.e. index==row*columns + column
   * columnOf(index)==index%columns rowOf(index)==index/columns i.e. {row0 column0..m}, {row1 column0..m}, ..., {rown
   * column0..m}
   */
  protected final Object[] elements;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  public DenseObjectMatrix2D(Object[][] values) {
    this(values.length, values.length == 0 ? 0 : values[0].length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of rows and columns. All entries are initially <tt>0</tt>.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (Object)columns*rows > Integer.MAX_VALUE</tt>.
   */
  public DenseObjectMatrix2D(int rows, int columns) {
    setUp(rows, columns);
    this.elements = new Object[rows * columns];
  }

  /**
   * Constructs a view with the given parameters.
   *
   * @param rows         the number of rows the matrix shall have.
   * @param columns      the number of columns the matrix shall have.
   * @param elements     the cells.
   * @param rowZero      the position of the first element.
   * @param columnZero   the position of the first element.
   * @param rowStride    the number of elements between two rows, i.e. <tt>index(i+1,j)-index(i,j)</tt>.
   * @param columnStride the number of elements between two columns, i.e. <tt>index(i,j+1)-index(i,j)</tt>.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (Object)columns*rows > Integer.MAX_VALUE</tt> or
   *                                  flip's are illegal.
   */
  protected DenseObjectMatrix2D(int rows, int columns, Object[] elements, int rowZero, int columnZero, int rowStride,
                                int columnStride) {
    setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
    this.elements = elements;
    this.isNoView = false;
  }

  /**
   * Sets all cells to the state specified by <tt>values</tt>. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of rows and columns as the receiver. <p> The values
   * are copied. So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values the values to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>values.length != rows() || for any 0 &lt;= row &lt; rows():
   *                                  values[row].length != columns()</tt>.
   */
  @Override
  public ObjectMatrix2D assign(Object[][] values) {
    if (this.isNoView) {
      if (values.length != rows) {
        throw new IllegalArgumentException("Must have same number of rows: rows=" + values.length + "rows()=" + rows());
      }
      int i = columns * (rows - 1);
      for (int row = rows; --row >= 0;) {
        Object[] currentRow = values[row];
        if (currentRow.length != columns) {
          throw new IllegalArgumentException(
              "Must have same number of columns in every row: columns=" + currentRow.length + "columns()=" + columns());
        }
        System.arraycopy(currentRow, 0, this.elements, i, columns);
        i -= columns;
      }
    } else {
      super.assign(values);
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
   * @return <tt>this</tt> (for convenience only).
   * @see org.apache.mahout.jet.math.Functions
   */
  @Override
  public ObjectMatrix2D assign(ObjectFunction function) {
    Object[] elems = this.elements;
    if (elems == null) {
      throw new InternalError();
    }
    int index = index(0, 0);
    int cs = this.columnStride;
    int rs = this.rowStride;

    // the general case x[i] = f(x[i])
    for (int row = rows; --row >= 0;) {
      for (int i = index, column = columns; --column >= 0;) {
        elems[i] = function.apply(elems[i]);
        i += cs;
      }
      index += rs;
    }
    return this;
  }

  /**
   * Replaces all cell values of the receiver with the values of another matrix. Both matrices must have the same number
   * of rows and columns. If both matrices share the same cells (as is the case if they are views derived from the same
   * matrix) and intersect in an ambiguous way, then replaces <i>as if</i> using an intermediate auxiliary deep copy of
   * <tt>other</tt>.
   *
   * @param source the source matrix to copy from (may be identical to the receiver).
   * @return <tt>this</tt> (for convenience only).
   * @throws IllegalArgumentException if <tt>columns() != source.columns() || rows() != source.rows()</tt>
   */
  @Override
  public ObjectMatrix2D assign(ObjectMatrix2D source) {
    // overriden for performance only
    if (!(source instanceof DenseObjectMatrix2D)) {
      return super.assign(source);
    }
    DenseObjectMatrix2D other = (DenseObjectMatrix2D) source;
    if (other == this) {
      return this;
    } // nothing to do
    checkShape(other);

    if (this.isNoView && other.isNoView) { // quickest
      System.arraycopy(other.elements, 0, this.elements, 0, this.elements.length);
      return this;
    }

    if (haveSharedCells(other)) {
      ObjectMatrix2D c = other.copy();
      if (!(c instanceof DenseObjectMatrix2D)) { // should not happen
        return super.assign(other);
      }
      other = (DenseObjectMatrix2D) c;
    }

    Object[] elems = this.elements;
    Object[] otherElems = other.elements;
    if (elements == null || otherElems == null) {
      throw new InternalError();
    }
    int cs = this.columnStride;
    int ocs = other.columnStride;
    int rs = this.rowStride;
    int ors = other.rowStride;

    int otherIndex = other.index(0, 0);
    int index = index(0, 0);
    for (int row = rows; --row >= 0;) {
      for (int i = index, j = otherIndex, column = columns; --column >= 0;) {
        elems[i] = otherElems[j];
        i += cs;
        j += ocs;
      }
      index += rs;
      otherIndex += ors;
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
   * m1.assign(m2, org.apache.mahout.jet.math.Functions.pow);
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
   * @see org.apache.mahout.jet.math.Functions
   */
  @Override
  public ObjectMatrix2D assign(ObjectMatrix2D y, org.apache.mahout.matrix.function.ObjectObjectFunction function) {
    // overriden for performance only
    if (!(y instanceof DenseObjectMatrix2D)) {
      return super.assign(y, function);
    }
    DenseObjectMatrix2D other = (DenseObjectMatrix2D) y;
    checkShape(y);

    Object[] elems = this.elements;
    Object[] otherElems = other.elements;
    if (elems == null || otherElems == null) {
      throw new InternalError();
    }
    int cs = this.columnStride;
    int ocs = other.columnStride;
    int rs = this.rowStride;
    int ors = other.rowStride;

    int otherIndex = other.index(0, 0);
    int index = index(0, 0);

    // the general case x[i] = f(x[i],y[i])
    for (int row = rows; --row >= 0;) {
      for (int i = index, j = otherIndex, column = columns; --column >= 0;) {
        elems[i] = function.apply(elems[i], otherElems[j]);
        i += cs;
        j += ocs;
      }
      index += rs;
      otherIndex += ors;
    }
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
  public Object getQuick(int row, int column) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows) throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //return elements[index(row,column)];
    //manually inlined:
    return elements[rowZero + row * rowStride + columnZero + column * columnStride];
  }

  /**
   * Returns <tt>true</tt> if both matrices share common cells. More formally, returns <tt>true</tt> if <tt>other !=
   * null</tt> and at least one of the following conditions is met <ul> <li>the receiver is a view of the other matrix
   * <li>the other matrix is a view of the receiver <li><tt>this == other</tt> </ul>
   */
  @Override
  protected boolean haveSharedCellsRaw(ObjectMatrix2D other) {
    if (other instanceof SelectedDenseObjectMatrix2D) {
      SelectedDenseObjectMatrix2D otherMatrix = (SelectedDenseObjectMatrix2D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof DenseObjectMatrix2D) {
      DenseObjectMatrix2D otherMatrix = (DenseObjectMatrix2D) other;
      return this.elements == otherMatrix.elements;
    }
    return false;
  }

  /**
   * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   */
  @Override
  protected int index(int row, int column) {
    // return super.index(row,column);
    // manually inlined for speed:
    return rowZero + row * rowStride + columnZero + column * columnStride;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified
   * number of rows and columns. For example, if the receiver is an instance of type <tt>DenseObjectMatrix2D</tt> the
   * new matrix must also be of type <tt>DenseObjectMatrix2D</tt>, if the receiver is an instance of type
   * <tt>SparseObjectMatrix2D</tt> the new matrix must also be of type <tt>SparseObjectMatrix2D</tt>, etc. In general,
   * the new matrix should have internal parametrization as similar as possible.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public ObjectMatrix2D like(int rows, int columns) {
    return new DenseObjectMatrix2D(rows, columns);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the
   * receiver. For example, if the receiver is an instance of type <tt>DenseObjectMatrix2D</tt> the new matrix must be
   * of type <tt>DenseObjectMatrix1D</tt>, if the receiver is an instance of type <tt>SparseObjectMatrix2D</tt> the new
   * matrix must be of type <tt>SparseObjectMatrix1D</tt>, etc.
   *
   * @param size the number of cells the matrix shall have.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  public ObjectMatrix1D like1D(int size) {
    return new DenseObjectMatrix1D(size);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseObjectMatrix2D</tt> the new matrix must be of type
   * <tt>DenseObjectMatrix1D</tt>, if the receiver is an instance of type <tt>SparseObjectMatrix2D</tt> the new matrix
   * must be of type <tt>SparseObjectMatrix1D</tt>, etc.
   *
   * @param size   the number of cells the matrix shall have.
   * @param zero   the index of the first element.
   * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  protected ObjectMatrix1D like1D(int size, int zero, int stride) {
    return new DenseObjectMatrix1D(size, this.elements, zero, stride);
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
  public void setQuick(int row, int column, Object value) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows) throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //elements[index(row,column)] = value;
    //manually inlined:
    elements[rowZero + row * rowStride + columnZero + column * columnStride] = value;
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected ObjectMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
    return new SelectedDenseObjectMatrix2D(this.elements, rowOffsets, columnOffsets, 0);
  }
}
