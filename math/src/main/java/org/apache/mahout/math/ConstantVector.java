package org.apache.mahout.math;

import com.google.common.collect.AbstractIterator;

import java.util.Iterator;

/**
 * Implements a vector with all the same values.
 */
public class ConstantVector extends AbstractVector {
  private double value;

  public ConstantVector(double value, int size) {
    super(size);
    this.value = value;
  }

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   *
   * @param rows    the row cardinality
   * @param columns the column cardinality
   * @return a Matrix
   */
  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  /**
   * @return true iff this implementation should be considered dense -- that it explicitly represents
   *         every value
   */
  @Override
  public boolean isDense() {
    return true;
  }

  /**
   * @return true iff this implementation should be considered to be iterable in index order in an
   *         efficient way. In particular this implies that {@link #iterator()} and {@link
   *         #iterateNonZero()} return elements in ascending order by index.
   */
  @Override
  public boolean isSequentialAccess() {
    return true;
  }

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element returned
   * for performance reasons, so if you need a copy of it, you should call {@link #getElement(int)}
   * for the given index
   *
   * @return An {@link java.util.Iterator} over all elements
   */
  @Override
  public Iterator<Element> iterator() {
    return new AbstractIterator<Element>() {
      int i = 0;
      int n = size();
      @Override
      protected Element computeNext() {
        if (i < n) {
          return new LocalElement(i++);
        } else {
          return endOfData();
        }
      }
    };
  }

  /**
   * Iterates over all non-zero elements. <p/> NOTE: Implementations may choose to reuse the Element
   * returned for performance reasons, so if you need a copy of it, you should call {@link
   * #getElement(int)} for the given index
   *
   * @return An {@link java.util.Iterator} over all non-zero elements
   */
  @Override
  public Iterator<Element> iterateNonZero() {
    return iterator();
  }

  /**
   * Return the value at the given index, without checking bounds
   *
   * @param index an int index
   * @return the double at the index
   */
  @Override
  public double getQuick(int index) {
    return value;
  }

  /**
   * Return an empty vector of the same underlying class as the receiver
   *
   * @return a Vector
   */
  @Override
  public Vector like() {
    return new DenseVector(size());
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  @Override
  public void setQuick(int index, double value) {
    throw new UnsupportedOperationException("Can't set a value in a constant matrix");
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int
   */
  @Override
  public int getNumNondefaultElements() {
    return size();
  }
}
