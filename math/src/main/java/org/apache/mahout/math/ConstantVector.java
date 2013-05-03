/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math;

import java.util.Iterator;

import com.google.common.collect.AbstractIterator;

/**
 * Implements a vector with all the same values.
 */
public class ConstantVector extends AbstractVector {
  private final double value;

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
   * Used internally by assign() to update multiple indices and values at once.
   * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
   * <p/>
   * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
   *
   * @param updates a mapping of indices to values to merge in the vector.
   */
  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    throw new UnsupportedOperationException("Cannot mutate a ConstantVector");
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
      private int i = 0;
      private final int n = size();
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

  @Override
  public double getLookupCost() {
    return 1;
  }

  @Override
  public double getIteratorAdvanceCost() {
    return 1;
  }

  @Override
  public boolean isAddConstantTime() {
    throw new UnsupportedOperationException("Cannot mutate a ConstantVector");
  }
}
