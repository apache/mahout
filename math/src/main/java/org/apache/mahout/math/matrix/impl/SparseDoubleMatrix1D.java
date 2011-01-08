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
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public final class SparseDoubleMatrix1D extends DoubleMatrix1D {
  /*
   * The elements of the matrix.
   */
  final AbstractIntDoubleMap elements;

  /**
   * Constructs a matrix with a copy of the given values. The values are copied. So subsequent changes in
   * <tt>values</tt> are not reflected in the matrix, and vice-versa.
   *
   * @param values The values to be filled into the new matrix.
   */
  public SparseDoubleMatrix1D(double[] values) {
    this(values.length);
    assign(values);
  }

  /**
   * Constructs a matrix with a given number of cells. All entries are initially <tt>0</tt>.
   *
   * @param size the number of cells the matrix shall have.
   * @throws IllegalArgumentException if <tt>size<0</tt>.
   */
  public SparseDoubleMatrix1D(int size) {
    this(size, size / 1000, 0.2, 0.5);
  }

  /**
   * Constructs a matrix with a given number of parameters. All entries are initially <tt>0</tt>. For details related to
   * memory usage see {@link org.apache.mahout.math.map.OpenIntDoubleHashMap}.
   *
   * @param size            the number of cells the matrix shall have.
   * @param initialCapacity the initial capacity of the hash map. If not known, set <tt>initialCapacity=0</tt> or
   *                        small.
   * @param minLoadFactor   the minimum load factor of the hash map.
   * @param maxLoadFactor   the maximum load factor of the hash map.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   * @throws IllegalArgumentException if <tt>size<0</tt>.
   */
  public SparseDoubleMatrix1D(int size, int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    setUp(size);
    this.elements = new OpenIntDoubleHashMap(initialCapacity, minLoadFactor, maxLoadFactor);
  }

  /**
   * Constructs a matrix view with a given number of parameters.
   *
   * @param size     the number of cells the matrix shall have.
   * @param elements the cells.
   * @param offset   the index of the first element.
   * @param stride   the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @throws IllegalArgumentException if <tt>size<0</tt>.
   */
  SparseDoubleMatrix1D(int size, AbstractIntDoubleMap elements, int offset, int stride) {
    setUp(size, offset, stride);
    this.elements = elements;
    this.isNoView = false;
  }

  /**
   * Sets all cells to the state specified by <tt>value</tt>.
   *
   * @param value the value to be filled into the cells.
   */
  @Override
  public void assign(double value) {
    // overriden for performance only
    if (this.isNoView && value == 0) {
      this.elements.clear();
    } else {
      super.assign(value);
    }
  }

  /** Returns the number of cells having non-zero values. */
  @Override
  public int cardinality() {
    return this.isNoView ? this.elements.size() : super.cardinality();
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
   * Returns the matrix cell value at coordinate <tt>index</tt>.
   *
   * <p>Provided with invalid parameters this method may return invalid objects without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
   *
   * @param index the index of the cell.
   * @return the value of the specified cell.
   */
  @Override
  public double getQuick(int index) {
    //if (debug) if (index<0 || index>=size) checkIndex(index);
    //return this.elements.get(index(index));
    // manually inlined:
    return elements.get(zero + index * stride);
  }

  /** Returns <tt>true</tt> if both matrices share at least one identical cell. */
  @Override
  protected boolean haveSharedCellsRaw(DoubleMatrix1D other) {
    if (other instanceof SelectedSparseDoubleMatrix1D) {
      SelectedSparseDoubleMatrix1D otherMatrix = (SelectedSparseDoubleMatrix1D) other;
      return this.elements == otherMatrix.elements;
    } else if (other instanceof SparseDoubleMatrix1D) {
      SparseDoubleMatrix1D otherMatrix = (SparseDoubleMatrix1D) other;
      return this.elements == otherMatrix.elements;
    }
    return false;
  }

  /**
   * Returns the position of the element with the given relative rank within the (virtual or non-virtual) internal
   * 1-dimensional array. You may want to override this method for performance.
   *
   * @param rank the rank of the element.
   */
  @Override
  protected int index(int rank) {
    // overriden for manual inlining only
    //return _offset(_rank(rank));
    return zero + rank * stride;
  }

  /**
   * Construct and returns a new empty matrix <i>of the same dynamic type</i> as the receiver, having the specified
   * size. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix1D</tt> the new matrix must also be
   * of type <tt>DenseDoubleMatrix1D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix1D</tt> the new
   * matrix must also be of type <tt>SparseDoubleMatrix1D</tt>, etc. In general, the new matrix should have internal
   * parametrization as similar as possible.
   *
   * @param size the number of cell the matrix shall have.
   * @return a new empty matrix of the same dynamic type.
   */
  @Override
  public DoubleMatrix1D like(int size) {
    return new SparseDoubleMatrix1D(size);
  }

  /**
   * Construct and returns a new 2-d matrix <i>of the corresponding dynamic type</i>, entirelly independent of the
   * receiver. For example, if the receiver is an instance of type <tt>DenseDoubleMatrix1D</tt> the new matrix must be
   * of type <tt>DenseDoubleMatrix2D</tt>, if the receiver is an instance of type <tt>SparseDoubleMatrix1D</tt> the new
   * matrix must be of type <tt>SparseDoubleMatrix2D</tt>, etc.
   *
   * @param rows    the number of rows the matrix shall have.
   * @param columns the number of columns the matrix shall have.
   * @return a new matrix of the corresponding dynamic type.
   */
  @Override
  public DoubleMatrix2D like2D(int rows, int columns) {
    return new SparseDoubleMatrix2D(rows, columns);
  }

  /**
   * Sets the matrix cell at coordinate <tt>index</tt> to the specified value.
   *
   * <p>Provided with invalid parameters this method may access illegal indexes without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the coordinate is within bounds.</b> Precondition
   * (unchecked): <tt>index&lt;0 || index&gt;=size()</tt>.
   *
   * @param index the index of the cell.
   * @param value the value to be filled into the specified cell.
   */
  @Override
  public void setQuick(int index, double value) {
    //if (debug) if (index<0 || index>=size) checkIndex(index);
    //int i =  index(index);
    // manually inlined:
    int i = zero + index * stride;
    if (value == 0) {
      this.elements.removeKey(i);
    } else {
      this.elements.put(i, value);
    }
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param offsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
    return new SelectedSparseDoubleMatrix1D(this.elements, offsets);
  }
}
