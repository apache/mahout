/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.matrix.impl;

import org.apache.mahout.matrix.matrix.ObjectMatrix2D;
import org.apache.mahout.matrix.matrix.ObjectMatrix3D;
/**
 Dense 3-d matrix holding <tt>Object</tt> elements.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Implementation:</b>
 <p>
 Internally holds one single contigous one-dimensional array, addressed in (in decreasing order of significance): slice major, row major, column major.
 Note that this implementation is not synchronized.
 <p>
 <b>Memory requirements:</b>
 <p>
 <tt>memory [bytes] = 8*slices()*rows()*columns()</tt>.
 Thus, a 100*100*100 matrix uses 8 MB.
 <p>
 <b>Time complexity:</b>
 <p>
 <tt>O(1)</tt> (i.e. constant time) for the basic operations
 <tt>get</tt>, <tt>getQuick</tt>, <tt>set</tt>, <tt>setQuick</tt> and <tt>size</tt>,
 <p>
 Applications demanding utmost speed can exploit knowledge about the internal addressing.
 Setting/getting values in a loop slice-by-slice, row-by-row, column-by-column is quicker than, for example, column-by-column, row-by-row, slice-by-slice.
 Thus
 <pre>
 for (int slice=0; slice < slices; slice++) {
 for (int row=0; row < rows; row++) {
 for (int column=0; column < columns; column++) {
 matrix.setQuick(slice,row,column,someValue);
 }
 }
 }
 </pre>
 is quicker than
 <pre>
 for (int column=0; column < columns; column++) {
 for (int row=0; row < rows; row++) {
 for (int slice=0; slice < slices; slice++) {
 matrix.setQuick(slice,row,column,someValue);
 }
 }
 }
 </pre>
 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DenseObjectMatrix3D extends ObjectMatrix3D {

  /**
   * The elements of this matrix. elements are stored in slice major, then row major, then column major, in order of
   * significance, i.e. index==slice*sliceStride+ row*rowStride + column*columnStride i.e. {slice0 row0..m}, {slice1
   * row0..m}, ..., {sliceN row0..m} with each row storead as {row0 column0..m}, {row1 column0..m}, ..., {rown
   * column0..m}
   */
  protected Object[] elements;

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
  public DenseObjectMatrix3D(Object[][][] values) {
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
   * @throws IllegalArgumentException if <tt>(Object)slices*columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  public DenseObjectMatrix3D(int slices, int rows, int columns) {
    setUp(slices, rows, columns);
    this.elements = new Object[slices * rows * columns];
  }

  /**
   * Constructs a view with the given parameters.
   *
   * @param slices        the number of slices the matrix shall have.
   * @param rows          the number of rows the matrix shall have.
   * @param columns       the number of columns the matrix shall have.
   * @param elements      the cells.
   * @param sliceZero     the position of the first element.
   * @param rowZero       the position of the first element.
   * @param columnZero    the position of the first element.
   * @param sliceStride   the number of elements between two slices, i.e. <tt>index(k+1,i,j)-index(k,i,j)</tt>.
   * @param rowStride     the number of elements between two rows, i.e. <tt>index(k,i+1,j)-index(k,i,j)</tt>.
   * @param columnnStride the number of elements between two columns, i.e. <tt>index(k,i,j+1)-index(k,i,j)</tt>.
   * @throws IllegalArgumentException if <tt>(Object)slices*columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  protected DenseObjectMatrix3D(int slices, int rows, int columns, Object[] elements, int sliceZero, int rowZero,
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
  public ObjectMatrix3D assign(Object[][][] values) {
    if (this.isNoView) {
      if (values.length != slices) {
        throw new IllegalArgumentException(
            "Must have same number of slices: slices=" + values.length + "slices()=" + slices());
      }
      int i = slices * rows * columns - columns;
      for (int slice = slices; --slice >= 0;) {
        Object[][] currentSlice = values[slice];
        if (currentSlice.length != rows) {
          throw new IllegalArgumentException(
              "Must have same number of rows in every slice: rows=" + currentSlice.length + "rows()=" + rows());
        }
        for (int row = rows; --row >= 0;) {
          Object[] currentRow = currentSlice[row];
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
  public ObjectMatrix3D assign(ObjectMatrix3D source) {
    // overriden for performance only
    if (!(source instanceof DenseObjectMatrix3D)) {
      return super.assign(source);
    }
    DenseObjectMatrix3D other = (DenseObjectMatrix3D) source;
    if (other == this) {
      return this;
    }
    checkShape(other);
    if (haveSharedCells(other)) {
      ObjectMatrix3D c = other.copy();
      if (!(c instanceof DenseObjectMatrix3D)) { // should not happen
        return super.assign(source);
      }
      other = (DenseObjectMatrix3D) c;
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
  public Object getQuick(int slice, int row, int column) {
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
  protected boolean haveSharedCellsRaw(ObjectMatrix3D other) {
    if (other instanceof SelectedDenseObjectMatrix3D) {
      SelectedDenseObjectMatrix3D otherMatrix = (SelectedDenseObjectMatrix3D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof DenseObjectMatrix3D) {
      DenseObjectMatrix3D otherMatrix = (DenseObjectMatrix3D) other;
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
   * <tt>DenseObjectMatrix3D</tt> the new matrix must also be of type <tt>DenseObjectMatrix3D</tt>, if the receiver is
   * an instance of type <tt>SparseObjectMatrix3D</tt> the new matrix must also be of type
   * <tt>SparseObjectMatrix3D</tt>, etc. In general, the new matrix should have internal parametrization as similar as
   * possible.
   *
   * @param slices  the number of slices the matrix shall have.
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public ObjectMatrix3D like(int slices, int rows, int columns) {
    return new DenseObjectMatrix3D(slices, rows, columns);
  }

  /**
   * Construct and returns a new 2-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseObjectMatrix3D</tt> the new matrix must also be of type
   * <tt>DenseObjectMatrix2D</tt>, if the receiver is an instance of type <tt>SparseObjectMatrix3D</tt> the new matrix
   * must also be of type <tt>SparseObjectMatrix2D</tt>, etc.
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
  protected ObjectMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
    return new DenseObjectMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
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
  public void setQuick(int slice, int row, int column, Object value) {
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
  protected ObjectMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
    return new SelectedDenseObjectMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
  }
}
