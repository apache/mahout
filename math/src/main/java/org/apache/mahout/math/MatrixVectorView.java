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
  private boolean isDense = true;

  public MatrixVectorView(Matrix matrix, int row, int column, int rowStride, int columnStride, boolean isDense) {
    this(matrix, row, column, rowStride, columnStride);
    this.isDense = isDense;
  }

  public MatrixVectorView(Matrix matrix, int row, int column, int rowStride, int columnStride) {
    super(viewSize(matrix, row, column, rowStride, columnStride));
    if (row < 0 || row >= matrix.rowSize()) {
      throw new IndexException(row, matrix.rowSize());
    }
    if (column < 0 || column >= matrix.columnSize()) {
      throw new IndexException(column, matrix.columnSize());
    }

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
  @Override
  public boolean isDense() {
    return isDense;
  }

  /**
   * @return true iff {@link Vector} should be considered to be iterable in
   *         index order in an efficient way. In particular this implies that {@link #iterator()} and
   *         {@link #iterateNonZero()} return elements in ascending order by index.
   */
  @Override
  public boolean isSequentialAccess() {
    return true;
  }

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element returned
   * for performance reasons, so if you need a copy of it, you should call {@link #getElement(int)} for
   * the given index
   *
   * @return An {@link java.util.Iterator} over all elements
   */
  @Override
  public Iterator<Element> iterator() {
    final LocalElement r = new LocalElement(0);
    return new Iterator<Element>() {
      private int i;

      @Override
      public boolean hasNext() {
        return i < size();
      }

      @Override
      public Element next() {
        if (i >= size()) {
          throw new NoSuchElementException();
        }
        r.index = i++;
        return r;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException("Can't remove from a view");
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
    return matrix.getQuick(row + rowStride * index, column + columnStride * index);
  }

  /**
   * Return an empty vector of the same underlying class as the receiver
   *
   * @return a Vector
   */
  @Override
  public Vector like() {
    return matrix.like(size(), 1).viewColumn(0);
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  @Override
  public void setQuick(int index, double value) {
    matrix.setQuick(row + rowStride * index, column + columnStride * index, value);
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
    // TODO: what is a genuine value here?
    return 1;
  }

  @Override
  public double getIteratorAdvanceCost() {
    // TODO: what is a genuine value here?
    return 1;
  }

  @Override
  public boolean isAddConstantTime() {
    // TODO: what is a genuine value here?
    return true;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return matrix.like(rows, columns);
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
    int[] indices = updates.getIndices();
    double[] values = updates.getValues();
    for (int i = 0; i < updates.getNumMappings(); ++i) {
      matrix.setQuick(row + rowStride * indices[i], column + columnStride * indices[i], values[i]);
    }
  }
}
