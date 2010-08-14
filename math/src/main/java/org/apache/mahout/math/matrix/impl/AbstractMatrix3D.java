/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

/**
 Abstract base class for 3-d matrices holding objects or primitive data types such as <code>int</code>, <code>double</code>, etc.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Note that this implementation is not synchronized.</b>

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public abstract class AbstractMatrix3D extends AbstractMatrix {

  /** the number of slices this matrix (view) has */
  protected int slices;

  /** the number of rows this matrix (view) has */
  protected int rows;

  /** the number of columns this matrix (view) has */
  protected int columns;


  /** the number of elements between two slices, i.e. <tt>index(k+1,i,j) - index(k,i,j)</tt>. */
  protected int sliceStride;

  /** the number of elements between two rows, i.e. <tt>index(k,i+1,j) - index(k,i,j)</tt>. */
  protected int rowStride;

  /** the number of elements between two columns, i.e. <tt>index(k,i,j+1) - index(k,i,j)</tt>. */
  protected int columnStride;

  /** the index of the first element */
  protected int sliceZero;
  protected int rowZero;
  protected int columnZero;

  // this.isNoView implies: offset==0, sliceStride==rows*slices, rowStride==columns, columnStride==1

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractMatrix3D() {
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int _columnOffset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int _columnRank(int rank) {
    return columnZero + rank * columnStride;
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int _rowOffset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int _rowRank(int rank) {
    return rowZero + rank * rowStride;
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int _sliceOffset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int _sliceRank(int rank) {
    return sliceZero + rank * sliceStride;
  }

  /**
   * Checks whether the receiver contains the given box and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>row<0 || height<0 || row+height>rows || slice<0 || depth<0 ||
   *                                   slice+depth>slices  || column<0 || width<0 || column+width>columns</tt>
   */
  protected void checkBox(int slice, int row, int column, int depth, int height, int width) {
    if (slice < 0 || depth < 0 || slice + depth > slices || row < 0 || height < 0 || row + height > rows ||
        column < 0 || width < 0 || column + width > columns) {
      throw new IndexOutOfBoundsException(
          toStringShort() + ", slice:" + slice + ", row:" + row + " ,column:" + column + ", depth:" + depth +
              " ,height:" + height + ", width:" + width);
    }
  }

  /**
   * Sanity check for operations requiring a column index to be within bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= columns()</tt>.
   */
  protected void checkColumn(int column) {
    if (column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException("Attempted to access " + toStringShort() + " at column=" + column);
    }
  }

  /**
   * Checks whether indexes are legal and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>! (0 <= indexes[i] < columns())</tt> for any i=0..indexes.length()-1.
   */
  protected void checkColumnIndexes(int[] indexes) {
    for (int i = indexes.length; --i >= 0;) {
      int index = indexes[i];
      if (index < 0 || index >= columns) {
        checkColumn(index);
      }
    }
  }

  /**
   * Sanity check for operations requiring a row index to be within bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>row < 0 || row >= rows()</tt>.
   */
  protected void checkRow(int row) {
    if (row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("Attempted to access " + toStringShort() + " at row=" + row);
    }
  }

  /**
   * Checks whether indexes are legal and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>! (0 <= indexes[i] < rows())</tt> for any i=0..indexes.length()-1.
   */
  protected void checkRowIndexes(int[] indexes) {
    for (int i = indexes.length; --i >= 0;) {
      int index = indexes[i];
      if (index < 0 || index >= rows) {
        checkRow(index);
      }
    }
  }

  /**
   * Sanity check for operations requiring two matrices with the same number of slices, rows and columns.
   *
   * @throws IllegalArgumentException if <tt>slices() != B.slices() || rows() != B.rows() || columns() !=
   *                                  B.columns()</tt>.
   */
  public void checkShape(AbstractMatrix3D B) {
    if (slices != B.slices || rows != B.rows || columns != B.columns) {
      throw new IllegalArgumentException("Incompatible dimensions: " + toStringShort() + " and " + B.toStringShort());
    }
  }

  /**
   * Sanity check for operations requiring matrices with the same number of slices, rows and columns.
   *
   * @throws IllegalArgumentException if <tt>slices() != B.slices() || rows() != B.rows() || columns() != B.columns() ||
   *                                  slices() != C.slices() || rows() != C.rows() || columns() != C.columns()</tt>.
   */
  public void checkShape(AbstractMatrix3D B, AbstractMatrix3D C) {
    if (slices != B.slices || rows != B.rows || columns != B.columns || slices != C.slices || rows != C.rows ||
        columns != C.columns) {
      throw new IllegalArgumentException(
          "Incompatible dimensions: " + toStringShort() + ", " + B.toStringShort() + ", " + C.toStringShort());
    }
  }

  /**
   * Sanity check for operations requiring a slice index to be within bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>slice < 0 || slice >= slices()</tt>.
   */
  protected void checkSlice(int slice) {
    if (slice < 0 || slice >= slices) {
      throw new IndexOutOfBoundsException("Attempted to access " + toStringShort() + " at slice=" + slice);
    }
  }

  /**
   * Checks whether indexes are legal and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>! (0 <= indexes[i] < slices())</tt> for any i=0..indexes.length()-1.
   */
  protected void checkSliceIndexes(int[] indexes) {
    for (int i = indexes.length; --i >= 0;) {
      int index = indexes[i];
      if (index < 0 || index >= slices) {
        checkSlice(index);
      }
    }
  }

  /** Returns the number of columns. */
  public int columns() {
    return columns;
  }

  /**
   * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array.
   *
   * @param slice  the index of the slice-coordinate.
   * @param row    the index of the row-coordinate.
   * @param column the index of the third-coordinate.
   */
  protected int index(int slice, int row, int column) {
    return _sliceOffset(_sliceRank(slice)) + _rowOffset(_rowRank(row)) + _columnOffset(_columnRank(column));
  }

  /** Returns the number of rows. */
  public int rows() {
    return rows;
  }

  /**
   * Sets up a matrix with a given number of slices and rows.
   *
   * @param slices  the number of slices the matrix shall have.
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>(double)rows*slices > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  protected void setUp(int slices, int rows, int columns) {
    setUp(slices, rows, columns, 0, 0, 0, rows * columns, columns, 1);
  }

  /**
   * Sets up a matrix with a given number of slices and rows and the given strides.
   *
   * @param slices       the number of slices the matrix shall have.
   * @param rows         the number of rows the matrix shall have.
   * @param columns      the number of columns the matrix shall have.
   * @param sliceZero    the position of the first element.
   * @param rowZero      the position of the first element.
   * @param columnZero   the position of the first element.
   * @param sliceStride  the number of elements between two slices, i.e. <tt>index(k+1,i,j)-index(k,i,j)</tt>.
   * @param rowStride    the number of elements between two rows, i.e. <tt>index(k,i+1,j)-index(k,i,j)</tt>.
   * @param columnStride the number of elements between two columns, i.e. <tt>index(k,i,j+1)-index(k,i,j)</tt>.
   * @throws IllegalArgumentException if <tt>(double)slices*rows*columnss > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  protected void setUp(int slices, int rows, int columns, int sliceZero, int rowZero, int columnZero, int sliceStride,
                       int rowStride, int columnStride) {
    if (slices < 0 || rows < 0 || columns < 0) {
      throw new IllegalArgumentException("negative size");
    }
    if ((double) slices * rows * columns > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("matrix too large");
    }

    this.slices = slices;
    this.rows = rows;
    this.columns = columns;

    this.sliceZero = sliceZero;
    this.rowZero = rowZero;
    this.columnZero = columnZero;

    this.sliceStride = sliceStride;
    this.rowStride = rowStride;
    this.columnStride = columnStride;

    this.isNoView = true;
  }

  protected int[] shape() {
    int[] shape = new int[3];
    shape[0] = slices;
    shape[1] = rows;
    shape[2] = columns;
    return shape;
  }

  /** Returns the number of cells which is <tt>slices()*rows()*columns()</tt>. */
  @Override
  public int size() {
    return slices * rows * columns;
  }

  /** Returns the number of slices. */
  public int slices() {
    return slices;
  }

  /** Returns a string representation of the receiver's shape. */
  public String toStringShort() {
    return AbstractFormatter.shape(this);
  }

  /** Self modifying version of viewColumnFlip(). */
  protected AbstractMatrix3D vColumnFlip() {
    if (columns > 0) {
      columnZero += (columns - 1) * columnStride;
      columnStride = -columnStride;
      this.isNoView = false;
    }
    return this;
  }

  /**
   * Self modifying version of viewDice().
   *
   * @throws IllegalArgumentException if some of the parameters are equal or not in range 0..2.
   */
  protected AbstractMatrix3D vDice(int axis0, int axis1, int axis2) {
    int d = 3;
    if (axis0 < 0 || axis0 >= d || axis1 < 0 || axis1 >= d || axis2 < 0 || axis2 >= d ||
        axis0 == axis1 || axis0 == axis2 || axis1 == axis2) {
      throw new IllegalArgumentException("Illegal Axes: " + axis0 + ", " + axis1 + ", " + axis2);
    }

    // swap shape
    int[] shape = shape();

    this.slices = shape[axis0];
    this.rows = shape[axis1];
    this.columns = shape[axis2];

    // swap strides
    int[] strides = new int[3];
    strides[0] = this.sliceStride;
    strides[1] = this.rowStride;
    strides[2] = this.columnStride;

    this.sliceStride = strides[axis0];
    this.rowStride = strides[axis1];
    this.columnStride = strides[axis2];

    this.isNoView = false;
    return this;
  }

  /**
   * Self modifying version of viewPart().
   *
   * @throws IndexOutOfBoundsException if <tt>slice<0 || depth<0 || slice+depth>slices() || row<0 || height<0 ||
   *                                   row+height>rows() || column<0 || width<0 || column+width>columns()</tt>
   */
  protected AbstractMatrix3D vPart(int slice, int row, int column, int depth, int height, int width) {
    checkBox(slice, row, column, depth, height, width);

    this.sliceZero += this.sliceStride * slice;
    this.rowZero += this.rowStride * row;
    this.columnZero += this.columnStride * column;

    this.slices = depth;
    this.rows = height;
    this.columns = width;

    this.isNoView = false;
    return this;
  }

  /** Self modifying version of viewRowFlip(). */
  protected AbstractMatrix3D vRowFlip() {
    if (rows > 0) {
      rowZero += (rows - 1) * rowStride;
      rowStride = -rowStride;
      this.isNoView = false;
    }
    return this;
  }

  /** Self modifying version of viewSliceFlip(). */
  protected AbstractMatrix3D vSliceFlip() {
    if (slices > 0) {
      sliceZero += (slices - 1) * sliceStride;
      sliceStride = -sliceStride;
      this.isNoView = false;
    }
    return this;
  }

  /**
   * Self modifying version of viewStrides().
   *
   * @throws IndexOutOfBoundsException if <tt>sliceStride<=0 || rowStride<=0 || columnStride<=0</tt>.
   */
  protected AbstractMatrix3D vStrides(int sliceStride, int rowStride, int columnStride) {
    if (sliceStride <= 0 || rowStride <= 0 || columnStride <= 0) {
      throw new IndexOutOfBoundsException("illegal strides: " + sliceStride + ", " + rowStride + ", " + columnStride);
    }

    this.sliceStride *= sliceStride;
    this.rowStride *= rowStride;
    this.columnStride *= columnStride;

    if (this.slices != 0) {
      this.slices = (this.slices - 1) / sliceStride + 1;
    }
    if (this.rows != 0) {
      this.rows = (this.rows - 1) / rowStride + 1;
    }
    if (this.columns != 0) {
      this.columns = (this.columns - 1) / columnStride + 1;
    }

    this.isNoView = false;
    return this;
  }
}
