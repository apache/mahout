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
 Abstract base class for 2-d matrices holding objects or primitive data types such as
 {@code int}, {@code double}, etc.
 First see the <a href="package-summary.html">package summary</a> and javadoc
 <a href="package-tree.html">tree view</a> to get the broad picture.
 <p>
 <b>Note that this implementation is not synchronized.</b>

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 09/24/99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public abstract class AbstractMatrix2D extends AbstractMatrix {

  /** the number of colums and rows this matrix (view) has */
  protected int columns;
  protected int rows;

  /** the number of elements between two rows, i.e. <tt>index(i+1,j,k) - index(i,j,k)</tt>. */
  protected int rowStride;

  /** the number of elements between two columns, i.e. <tt>index(i,j+1,k) - index(i,j,k)</tt>. */
  protected int columnStride;

  /** the index of the first element */
  protected int rowZero;
  protected int columnZero;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractMatrix2D() {
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int columnOffset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int columnRank(int rank) {
    return columnZero + rank * columnStride;
    //return columnZero + ((rank+columnFlipMask)^columnFlipMask);
    //return columnZero + rank*columnFlip; // slower
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int rowOffset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int rowRank(int rank) {
    return rowZero + rank * rowStride;
    //return rowZero + ((rank+rowFlipMask)^rowFlipMask);
    //return rowZero + rank*rowFlip; // slower
  }

  /**
   * Checks whether the receiver contains the given box and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>column<0 || width<0 || column+width>columns() || row<0 || height<0 ||
   *                                   row+height>rows()</tt>
   */
  protected void checkBox(int row, int column, int height, int width) {
    if (column < 0 || width < 0 || column + width > columns || row < 0 || height < 0 || row + height > rows) {
      throw new IndexOutOfBoundsException(
          "Column:" + column + ", row:" + row + " ,width:" + width + ", height:" + height);
    }
  }

  /**
   * Sanity check for operations requiring a column index to be within bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>column < 0 || column >= columns()</tt>.
   */
  protected void checkColumn(int column) {
    if (column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException("Attempted to access at column=" + column);
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
      throw new IndexOutOfBoundsException("Attempted to access at row=" + row);
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
   * Sanity check for operations requiring two matrices with the same number of columns and rows.
   *
   * @throws IllegalArgumentException if <tt>columns() != B.columns() || rows() != B.rows()</tt>.
   */
  public void checkShape(AbstractMatrix2D B) {
    if (columns != B.columns || rows != B.rows) {
      throw new IllegalArgumentException("Incompatible dimensions");
    }
  }

  /** Returns the number of columns. */
  public int columns() {
    return columns;
  }

  /**
   * Returns the position of the given coordinate within the (virtual or non-virtual) internal 1-dimensional array.
   *
   * @param row    the index of the row-coordinate.
   * @param column the index of the column-coordinate.
   */
  protected int index(int row, int column) {
    return rowOffset(rowRank(row)) + columnOffset(columnRank(column));
  }

  /** Returns the number of rows. */
  public int rows() {
    return rows;
  }

  /**
   * Sets up a matrix with a given number of rows and columns.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  protected void setUp(int rows, int columns) {
    setUp(rows, columns, 0, 0, columns, 1);
  }

  /**
   * Sets up a matrix with a given number of rows and columns and the given strides.
   *
   * @param rows         the number of rows the matrix shall have.
   * @param columns      the number of columns the matrix shall have.
   * @param rowZero      the position of the first element.
   * @param columnZero   the position of the first element.
   * @param rowStride    the number of elements between two rows, i.e. <tt>index(i+1,j)-index(i,j)</tt>.
   * @param columnStride the number of elements between two columns, i.e. <tt>index(i,j+1)-index(i,j)</tt>.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt> or
   *                                  flip's are illegal.
   */
  protected void setUp(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
    if (rows < 0 || columns < 0) {
      throw new IllegalArgumentException("negative size");
    }
    this.rows = rows;
    this.columns = columns;

    this.rowZero = rowZero;
    this.columnZero = columnZero;

    this.rowStride = rowStride;
    this.columnStride = columnStride;

    this.isNoView = true;
    if ((double) columns * rows > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("matrix too large");
    }
  }

  /** Returns the number of cells which is <tt>rows()*columns()</tt>. */
  @Override
  public int size() {
    return rows * columns;
  }

  /** Self modifying version of viewColumnFlip(). */
  protected AbstractMatrix2D vColumnFlip() {
    if (columns > 0) {
      columnZero += (columns - 1) * columnStride;
      columnStride = -columnStride;
      this.isNoView = false;
    }
    return this;
  }

  /** Self modifying version of viewDice(). */
  protected AbstractMatrix2D vDice() {
    // swap;
    int tmp = rows;
    rows = columns;
    columns = tmp;
    tmp = rowStride;
    rowStride = columnStride;
    columnStride = tmp;
    tmp = rowZero;
    rowZero = columnZero;
    columnZero = tmp;

    // flips stay unaffected

    this.isNoView = false;
    return this;
  }

  /**
   * Self modifying version of viewPart().
   *
   * @throws IndexOutOfBoundsException if <tt>column<0 || width<0 || column+width>columns() || row<0 || height<0 ||
   *                                   row+height>rows()</tt>
   */
  protected AbstractMatrix2D vPart(int row, int column, int height, int width) {
    checkBox(row, column, height, width);
    this.rowZero += this.rowStride * row;
    this.columnZero += this.columnStride * column;
    this.rows = height;
    this.columns = width;
    this.isNoView = false;
    return this;
  }

  /** Self modifying version of viewRowFlip(). */
  protected AbstractMatrix2D vRowFlip() {
    if (rows > 0) {
      rowZero += (rows - 1) * rowStride;
      rowStride = -rowStride;
      this.isNoView = false;
    }
    return this;
  }

}
