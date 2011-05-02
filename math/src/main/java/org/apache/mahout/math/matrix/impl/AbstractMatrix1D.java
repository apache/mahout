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
 Abstract base class for 1-d matrices (aka <i>vectors</i>) holding objects or primitive data types such as
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
public abstract class AbstractMatrix1D extends AbstractMatrix {

  /** the number of cells this matrix (view) has */
  protected int size;
  /** the index of the first element */
  protected int zero;
  /** the number of indexes between any two elements, i.e. <tt>index(i+1) - index(i)</tt>. */
  protected int stride;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  protected AbstractMatrix1D() {
  }

  /**
   * Returns the position of the given absolute rank within the (virtual or non-virtual) internal 1-dimensional array.
   * Default implementation. Override, if necessary.
   *
   * @param absRank the absolute rank of the element.
   * @return the position.
   */
  protected int offset(int absRank) {
    return absRank;
  }

  /**
   * Returns the absolute rank of the given relative rank.
   *
   * @param rank the relative rank of the element.
   * @return the absolute rank of the element.
   */
  protected int rank(int rank) {
    return zero + rank * stride;
    //return zero + ((rank+flipMask)^flipMask);
    //return zero + rank*flip; // slower
  }

  /**
   * Sanity check for operations requiring an index to be within bounds.
   *
   * @throws IndexOutOfBoundsException if <tt>index < 0 || index >= size()</tt>.
   */
  protected void checkIndex(int index) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException("Attempted to access at index=" + index);
    }
  }

  /**
   * Checks whether the receiver contains the given range and throws an exception, if necessary.
   *
   * @throws IndexOutOfBoundsException if <tt>index<0 || index+width>size()</tt>.
   */
  protected void checkRange(int index, int width) {
    if (index < 0 || index + width > size) {
      throw new IndexOutOfBoundsException("index: " + index + ", width: " + width + ", size=" + size);
    }
  }

  /**
   * Sanity check for operations requiring two matrices with the same size.
   *
   * @throws IllegalArgumentException if <tt>size() != b.size()</tt>.
   */
  public void checkSize(AbstractMatrix1D b) {
    if (size != b.size) {
      throw new IllegalArgumentException("Incompatible sizes: " + size + " and " + b.size);
    }
  }

  /**
   * Returns the position of the element with the given relative rank within the (virtual or non-virtual) internal
   * 1-dimensional array. You may want to override this method for performance.
   *
   * @param rank the rank of the element.
   */
  protected int index(int rank) {
    return offset(rank(rank));
  }

  /**
   * Sets up a matrix with a given number of cells.
   *
   * @param size the number of cells the matrix shall have.
   * @throws IllegalArgumentException if <tt>size<0</tt>.
   */
  protected void setUp(int size) {
    setUp(size, 0, 1);
  }

  /**
   * Sets up a matrix with the given parameters.
   *
   * @param size   the number of elements the matrix shall have.
   * @param zero   the index of the first element.
   * @param stride the number of indexes between any two elements, i.e. <tt>index(i+1)-index(i)</tt>.
   * @throws IllegalArgumentException if <tt>size<0</tt>.
   */
  protected void setUp(int size, int zero, int stride) {
    if (size < 0) {
      throw new IllegalArgumentException("negative size");
    }

    this.size = size;
    this.zero = zero;
    this.stride = stride;
    this.isNoView = true;
  }

  /** Returns the number of cells. */
  @Override
  public int size() {
    return size;
  }

  /**
   * Returns the stride of the given dimension (axis, rank).
   *
   * @return the stride in the given dimension.
   * @throws IllegalArgumentException if <tt>dimension != 0</tt>.
   */
  protected int stride(int dimension) {
    if (dimension != 0) {
      throw new IllegalArgumentException("invalid dimension: " + dimension);
    }
    return this.stride;
  }

  /**
   * Self modifying version of viewPart().
   *
   * @throws IndexOutOfBoundsException if <tt>index<0 || index+width>size()</tt>.
   */
  protected AbstractMatrix1D vPart(int index, int width) {
    checkRange(index, width);
    this.zero += this.stride * index;
    this.size = width;
    this.isNoView = false;
    return this;
  }

}
