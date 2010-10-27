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
import java.util.NoSuchElementException;

/**
 * Provides a virtual vector that is really a row or column or diagonal of a matrix.
 */
public class MatrixVectorView extends AbstractVector {
  private Matrix matrix;
  private int row;
  private int column;
  private int rowStride;
  private int columnStride;

  public MatrixVectorView(Matrix matrix, int row, int column, int rowStride, int columnStride) {
    super(viewSize(matrix, row, column, rowStride, columnStride));
    this.matrix = matrix;
    this.row = row;
    this.column = column;
    this.rowStride = rowStride;
    this.columnStride = columnStride;
  }

  private static int viewSize(Matrix matrix, int row, int column, int rowStride, int columnStride) {
    if (rowStride != 0 && columnStride != 0) {
      int n1 = (matrix.numRows() - row) / rowStride;
      int n2 = (matrix.numCols() - column) / columnStride;
      return Math.min(n1, n2);
    } else if (rowStride > 0) {
      return (matrix.numRows() - row) / rowStride;
    } else {
      return (matrix.numCols() - column) / columnStride;
    }
  }

  /**
   * @return true iff the {@link Vector} implementation should be considered
   *         dense -- that it explicitly represents every value
   */
  public boolean isDense() {
    return true;
  }

  /**
   * @return true iff {@link Vector} should be considered to be iterable in
   *         index order in an efficient way. In particular this implies that {@link #iterator()} and
   *         {@link #iterateNonZero()} return elements in ascending order by index.
   */
  public boolean isSequentialAccess() {
    return true;
  }

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element returned
   * for performance reasons, so if you need a copy of it, you should call {@link #getElement} for
   * the given index
   *
   * @return An {@link java.util.Iterator} over all elements
   */
  public Iterator<Element> iterator() {
    final LocalElement r = new LocalElement(0);
    return new Iterator<Element>() {
      private int i;

      public boolean hasNext() {
        return i < size();
      }

      public Element next() {
        if (i >= size()) {
          throw new NoSuchElementException();
        }
        r.index = i++;
        return r;
      }

      public void remove() {
        throw new UnsupportedOperationException("Can't remove from a view");
      }
    };
  }

  /**
   * Iterates over all non-zero elements. <p/> NOTE: Implementations may choose to reuse the Element
   * returned for performance reasons, so if you need a copy of it, you should call {@link
   * #getElement} for the given index
   *
   * @return An {@link java.util.Iterator} over all non-zero elements
   */
  public Iterator<Element> iterateNonZero() {
    return iterator();
  }

  /**
   * Return the value at the given index, without checking bounds
   *
   * @param index an int index
   * @return the double at the index
   */
  public double getQuick(int index) {
    return matrix.getQuick(row + rowStride * index, column + columnStride * index);
  }

  /**
   * Return an empty vector of the same underlying class as the receiver
   *
   * @return a Vector
   */
  public Vector like() {
    return matrix.like(size(), 1).viewColumn(0);
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  public void setQuick(int index, double value) {
    matrix.setQuick(row + rowStride * index, column + columnStride * index, value);
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int
   */
  public int getNumNondefaultElements() {
    return size();
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
    int[] offset = {row, column};
    int[] size = {rowStride == 0 ? 1 : rowStride, columnStride == 0 ? 1 : columnStride};
    return matrix.viewPart(offset, size);
  }

  @Override
  public Vector clone() {
    MatrixVectorView r = (MatrixVectorView) super.clone();
    r.matrix = matrix.clone();
    r.row = row;
    r.column = column;
    r.rowStride = rowStride;
    r.columnStride = columnStride;
    return r;
  }
}
