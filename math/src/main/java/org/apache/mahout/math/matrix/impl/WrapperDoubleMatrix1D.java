/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.matrix.impl;

import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.DoubleMatrix2D;

/**
 * 1-d matrix holding <tt>double</tt> elements; either a view wrapping another matrix or a matrix whose views are
 * wrappers.
 *
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 09/24/99
 */
class WrapperDoubleMatrix1D extends DoubleMatrix1D {
  /*
   * The elements of the matrix.
   */
  private final DoubleMatrix1D content;

  WrapperDoubleMatrix1D(DoubleMatrix1D newContent) {
    if (newContent != null) {
      setUp(newContent.size());
    }
    this.content = newContent;
  }

  /**
   * Returns the content of this matrix if it is a wrapper; or <tt>this</tt> otherwise. Override this method in
   * wrappers.
   */
  @Override
  protected DoubleMatrix1D getContent() {
    return this.content;
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
    return content.getQuick(index);
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
    return content.like(size);
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
    return content.like2D(rows, columns);
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
    content.setQuick(index, value);
  }

  /**
   * Constructs and returns a new <i>sub-range view</i> that is a <tt>width</tt> sub matrix starting at <tt>index</tt>.
   *
   * Operations on the returned view can only be applied to the restricted range. Any attempt to access coordinates not
   * contained in the view will throw an <tt>IndexOutOfBoundsException</tt>. <p> <b>Note that the view is really just a
   * range restriction:</b> The returned matrix is backed by this matrix, so changes in the returned matrix are
   * reflected in this matrix, and vice-versa. <p> The view contains the cells from <tt>index..index+width-1</tt>. and
   * has <tt>view.size() == width</tt>. A view's legal coordinates are again zero based, as usual. In other words, legal
   * coordinates of the view are <tt>0 .. view.size()-1==width-1</tt>. As usual, any attempt to access a cell at other
   * coordinates will throw an <tt>IndexOutOfBoundsException</tt>.
   *
   * @param index The index of the first cell.
   * @param width The width of the range.
   * @return the new view.
   * @throws IndexOutOfBoundsException if <tt>index<0 || width<0 || index+width>size()</tt>.
   */
  @Override
  public DoubleMatrix1D viewPart(final int index, int width) {
    checkRange(index, width);
    DoubleMatrix1D view = new WrapperDoubleMatrix1D(this) {
      @Override
      public double getQuick(int i) {
        return content.get(index + i);
      }

      @Override
      public void setQuick(int i, double value) {
        content.set(index + i, value);
      }
    };
    view.size = width;
    return view;
  }

  /**
   * Construct and returns a new selection view.
   *
   * @param offsets the offsets of the visible elements.
   * @return a new view.
   */
  @Override
  protected DoubleMatrix1D viewSelectionLike(int[] offsets) {
    throw new UnsupportedOperationException(); // should never get called
  }

}
