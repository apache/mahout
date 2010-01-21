/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.map.AbstractIntDoubleMap;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.matrix.DoubleMatrix2D;
import org.apache.mahout.math.matrix.DoubleMatrix3D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class SparseDoubleMatrix3D extends DoubleMatrix3D {
  /*
   * The elements of the matrix.
   */
  protected final AbstractIntDoubleMap elements;

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
  public SparseDoubleMatrix3D(double[][][] values) {
    this(values.length, (values.length == 0 ? 0 : values[0].length),
        (values.length == 0 ? 0 : values[0].length == 0 ? 0 : values[0][0].length));
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of slices, rows and columns and default memory usage. All entries are
   * initially <tt>0</tt>.
   *
   * @param slices  the number of slices the matrix shall have.
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @throws IllegalArgumentException if <tt>(double)slices*columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  public SparseDoubleMatrix3D(int slices, int rows, int columns) {
    this(slices, rows, columns, slices * rows * (columns / 1000), 0.2, 0.5);
  }

  /**
   * Constructs a matrix with a given number of slices, rows and columns using memory as specified. All entries are
   * initially <tt>0</tt>. For details related to memory usage see {@link org.apache.mahout.math.map.OpenIntDoubleHashMap}.
   *
   * @param slices          the number of slices the matrix shall have.
   * @param rows            the number of rows the matrix shall have.
   * @param columns         the number of columns the matrix shall have.
   * @param initialCapacity the initial capacity of the hash map. If not known, set <tt>initialCapacity=0</tt> or
   *                        small.
   * @param minLoadFactor   the minimum load factor of the hash map.
   * @param maxLoadFactor   the maximum load factor of the hash map.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   * @throws IllegalArgumentException if <tt>(double)columns*rows > Integer.MAX_VALUE</tt>.
   * @throws IllegalArgumentException if <tt>slices<0 || rows<0 || columns<0</tt>.
   */
  public SparseDoubleMatrix3D(int slices, int rows, int columns, int initialCapacity, double minLoadFactor,
                              double maxLoadFactor) {
    setUp(slices, rows, columns);
    this.elements = new OpenIntDoubleHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
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
  protected SparseDoubleMatrix3D(int slices, int rows, int columns, AbstractIntDoubleMap elements, int sliceZero,
                                 int rowZero, int columnZero, int sliceStride, int rowStride, int columnStride) {
    setUp(slices, rows, columns, sliceZero, rowZero, columnZero, sliceStride, rowStride, columnStride);
    this.elements = elements;
    this.isNoView = false;
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   * @return <tt>this</tt> (for convenience only).
   */
  @Override
  public DoubleMatrix3D assign(double value) {
    // overriden for performance only
    if (this.isNoView && value == 0) {
      this.elements.clear();
    } else {
      super.assign(value);
    }
    return this;
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
    //return elements.get(index(slice,row,column));
    //manually inlined:
    return elements
        .get(sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride);
  }

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  @Override
  protected boolean haveSharedCellsRaw(DoubleMatrix3D other) {
    if (other instanceof SelectedSparseDoubleMatrix3D) {
      SelectedSparseDoubleMatrix3D otherMatrix = (SelectedSparseDoubleMatrix3D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof SparseDoubleMatrix3D) {
      SparseDoubleMatrix3D otherMatrix = (SparseDoubleMatrix3D) other;
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
    return new SparseDoubleMatrix3D(slices, rows, columns);
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
    return new SparseDoubleMatrix2D(rows, columns, this.elements, rowZero, columnZero, rowStride, columnStride);
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
    //int index =  index(slice,row,column);
    //manually inlined:
    int index = sliceZero + slice * sliceStride + rowZero + row * rowStride + columnZero + column * columnStride;
    if (value == 0) {
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
   * reclaimed by calling <tt>trimToSize()</tt>. </ul> A sequence like <tt>set(s,r,c,5); set(s,r,c,0);</tt> sets a cell
   * to non-zero state and later back to zero state. Such as sequence generates obsolete memory that is automatically
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
   * @param sliceOffsets  the offsets of the visible elements.
   * @param rowOffsets    the offsets of the visible elements.
   * @param columnOffsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected DoubleMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
    return new SelectedSparseDoubleMatrix3D(this.elements, sliceOffsets, rowOffsets, columnOffsets, 0);
  }
}
