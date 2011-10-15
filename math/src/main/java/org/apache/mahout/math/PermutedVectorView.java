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

import com.google.common.collect.AbstractIterator;

import java.util.Iterator;

/**
 * Provides a permuted view of a vector.
 */
public class PermutedVectorView extends AbstractVector {
  private final Vector vector;            // the vector containing the data
  private final int[] pivot;              // convert from external index to internal
  private final int[] unpivot;            // convert from internal index to external

  public PermutedVectorView(Vector vector, int[] pivot, int[] unpivot) {
    super(vector.size());
    this.vector = vector;
    this.pivot = pivot;
    this.unpivot = unpivot;
  }

  public PermutedVectorView(Vector vector, int[] pivot) {
    this(vector, pivot, reversePivotPermutation(pivot));
  }

  private static int[] reversePivotPermutation(int[] pivot) {
    int[] unpivot1 = new int[pivot.length];
    for (int i = 0; i < pivot.length; i++) {
      unpivot1[pivot[i]] = i;
    }
    return unpivot1;
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
    if (vector.isDense()) {
      return new DenseMatrix(rows, columns);
    } else {
      return new SparseRowMatrix(rows, columns);
    }
  }

  /**
   * @return true iff this implementation should be considered dense -- that it explicitly
   *         represents every value
   */
  @Override
  public boolean isDense() {
    return vector.isDense();
  }

  /**
   * @return true iff this implementation should be considered to be iterable in index order in an
   *         efficient way. In particular this implies that {@link #iterator()} and {@link
   *         #iterateNonZero()} return elements in ascending order by index.
   */
  @Override
  public boolean isSequentialAccess() {
    return vector.isSequentialAccess();
  }

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element
   * returned for performance reasons, so if you need a copy of it, you should call {@link
   * #getElement(int)} for the given index
   *
   * @return An {@link java.util.Iterator} over all elements
   */
  @Override
  public Iterator<Element> iterator() {
    return new AbstractIterator<Element>() {
      private final Iterator<Element> i = vector.iterator();

      @Override
      protected Vector.Element computeNext() {
        if (i.hasNext()) {
          final Element x = i.next();
          return new Element() {
            private final int index = unpivot[x.index()];

            @Override
            public double get() {
              return x.get();
            }

            @Override
            public int index() {
              return index;
            }

            @Override
            public void set(double value) {
              x.set(value);
            }
          };
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
    return new AbstractIterator<Element>() {
      private final Iterator<Element> i = vector.iterateNonZero();

      @Override
      protected Vector.Element computeNext() {
        if (i.hasNext()) {
          final Element x = i.next();
          return new Element() {
            private final int index = unpivot[x.index()];

            @Override
            public double get() {
              return x.get();
            }

            @Override
            public int index() {
              return index;
            }

            @Override
            public void set(double value) {
              x.set(value);
            }
          };
        } else {
          return endOfData();
        }
      }
    };
  }

  /**
   * Return the value at the given index, without checking bounds
   *
   * @param index an int index
   * @return the double at the index
   */
  @Override
  public double getQuick(int index) {
    return vector.getQuick(pivot[index]);
  }

  /**
   * Return an empty vector of the same underlying class as the receiver
   *
   * @return a Vector
   */
  @Override
  public Vector like() {
    return vector.like();
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  @Override
  public void setQuick(int index, double value) {
    vector.setQuick(pivot[index], value);
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int
   */
  @Override
  public int getNumNondefaultElements() {
    return vector.getNumNondefaultElements();
  }
}
