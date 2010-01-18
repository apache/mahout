/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.map.AbstractIntObjectMap;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.apache.mahout.math.matrix.ObjectMatrix1D;
import org.apache.mahout.math.matrix.ObjectMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class SparseObjectMatrix2D<T> extends ObjectMatrix2D<T> {
  /*
   * The elements of the matrix.
   */
  protected final AbstractIntObjectMap<T> elements;

  /**
   * Constructs a matrix with a copy of the given values. <tt>values</tt> is required to have the form
   * <tt>values[row][column]</tt> and have exactly the same number of columns in every row. <p> The values are copied.
   * So subsequent changes in <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   * @throws IllegalArgumentException if <tt>for any 1 &lt;= row &lt; values.length: values[row].length !=
   *                                  values[row-1].length</tt>.
   */
  public SparseObjectMatrix2D(T[][] values) {
    this(values.length, values.length == 0 ? 0 : values[0].length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of rows and columns and default memory usage. All entries are initially
   * <tt>null</tt>.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  public SparseObjectMatrix2D(int rows, int columns) {
    this(rows, columns, rows * (columns / 1000), 0.2, 0.5);
  }

  /**
   * Constructs a matrix with a given number of rows and columns using memory as specified. All entries are initially
   * <tt>null</tt>. For details related to memory usage see {@link org.apache.mahout.math.map.OpenIntObjectHashMap}.
   *
   * @param rows            the number of rows the matrix shall have.
   * @param columns         the number of columns the matrix shall have.
   * @param initialCapacity the initial capacity of the hash map. If not known, set <tt>initialCapacity=0</tt> or
   *                        small.
   * @param minLoadFactor   the minimum load factor of the hash map.
   * @param maxLoadFactor   the maximum load factor of the hash map.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt>.
   */
  public SparseObjectMatrix2D(int rows, int columns, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    setUp(rows, columns);
    this.elements = new OpenIntObjectHashMap<T>(initialCapacity, minLoadFactor, maxLoadFactor);
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
   * @throws IllegalArgumentException if <tt>rows<0 || columns<0 || (double)columns*rows > Integer.MAX_VALUE</tt> or
   *                                  flip's are illegal.
   */
  protected SparseObjectMatrix2D(int rows, int columns, 
                                 AbstractIntObjectMap<T> elements, int rowZero, int columnZero,
                                 int rowStride, int columnStride) {
    setUp(rows, columns, rowZero, columnZero, rowStride, columnStride);
    this.elements = elements;
    this.isNoView = false;
  }

  /** Returns the number of cells having non-zero values. */
  @Override
  public int cardinality() {
    if (this.isNoView) {
      return this.elements.size();
    } else {
      return super.cardinality();
    }
  }

  /**
   * Ensures that the receiver can hold at least the specified number of non-zero cells without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver. <p> This
   * method never need be called; it is for performance tuning only. Calling this method before tt>set()</tt>ing a large
   * number of non-zero values boosts performance, because the receiver will grow only once instead of potentially many
   * times and hash collisions get less probable.
   *
   * @param minCapacity the desired minimum number of non-zero cells.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    this.elements.ensureCapacity(minCapacity);
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
  public T getQuick(int row, int column) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows) throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //return this.elements.get(index(row,column));
    //manually inlined:
    return this.elements.get(rowZero + row * rowStride + columnZero + column * columnStride);
  }

  /**
   * Returns <tt>true</tt> if both matrices share common cells. More formally, returns <tt>true</tt> if at least one of
   * the following conditions is met <ul> <li>the receiver is a view of the other matrix <li>the other matrix is a view
   * of the receiver <li><tt>this == other</tt> </ul>
   */
  @SuppressWarnings("unchecked")
  @Override
  protected boolean haveSharedCellsRaw(ObjectMatrix2D<T> other) {
    if (other instanceof SelectedSparseObjectMatrix2D) {
      SelectedSparseObjectMatrix2D otherMatrix = (SelectedSparseObjectMatrix2D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof SparseObjectMatrix2D) {
      SparseObjectMatrix2D otherMatrix = (SparseObjectMatrix2D) other;
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
  public ObjectMatrix2D<T> like(int rows, int columns) {
    return new SparseObjectMatrix2D<T>(rows, columns);
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
  public ObjectMatrix1D<T> like1D(int size) {
    return new SparseObjectMatrix1D<T>(size);
  }

  /**
   * Construct and returns a new 1-d matrix <i>of the corresponding dynamic type</i>, sharing the same cells. For
   * example, if the receiver is an instance of type <tt>DenseObjectMatrix2D</tt> the new matrix must be of type
   * <tt>DenseObjectMatrix1D</tt>, if the receiver is an instance of type <tt>SparseObjectMatrix2D</tt> the new matrix
   * must be of type <tt>SparseObjectMatrix1D</tt>, etc.
   *
   * @param size   the number of cells the matrix shall have.
   * @param offset the index of the first element.
   * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  protected ObjectMatrix1D<T> like1D(int size, int offset, int stride) {
    return new SparseObjectMatrix1D<T>(size, this.elements, offset, stride);
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
  public void setQuick(int row, int column, T value) {
    //if (debug) if (column<0 || column>=columns || row<0 || row>=rows) throw new IndexOutOfBoundsException("row:"+row+", column:"+column);
    //int index =  index(row,column);
    //manually inlined:
    int index = rowZero + row * rowStride + columnZero + column * columnStride;

    if (value == null) {
      this.elements.removeKey(index);
    } else {
      this.elements.put(index, value);
    }
  }

  /**
   * Releases any superfluous memory created by explicitly putting zero values into cells formerly having non-zero
   * values; An application can use this operation to minimize the storage of the receiver. <p> <b>Background:</b> <p>
   * Cells that <ul> <li>are never set to non-zero values do not use any memory. <li>switch from zero to non-zero state
   * do use memory. <li>switch back from non-zero to zero state also do use memory. However, their memory can be
   * reclaimed by calling <tt>trimToSize()</tt>. </ul> A sequence like <tt>set(r,c,5); set(r,c,0);</tt> sets a cell to
   * non-zero state and later back to zero state. Such as sequence generates obsolete memory that is automatically
   * reclaimed from time to time or can manually be reclaimed by calling <tt>trimToSize()</tt>. Putting zeros into cells
   * already containing zeros does not generate obsolete memory since no memory was allocated to them in the first
   * place.
   */
  @Override
  public void trimToSize() {
    this.elements.trimToSize();
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected ObjectMatrix2D<T> viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
    return new SelectedSparseObjectMatrix2D<T>(this.elements, rowOffsets, columnOffsets, 0);
  }
}
