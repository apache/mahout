/**
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

/**
 * sparse matrix with general element values whose columns are accessible quickly. Implemented as a column array of
 * SparseVectors.
 */
public class SparseColumnMatrix extends AbstractMatrix {

  private int[] cardinality;

  private Vector[] columns;

  public SparseColumnMatrix() {
  }

  /**
   * Construct a matrix of the given cardinality with the given data columns
   *
   * @param cardinality the int[2] cardinality
   * @param columns     a RandomAccessSparseVector[] array of columns
   */
  public SparseColumnMatrix(int[] cardinality, RandomAccessSparseVector[] columns) {
    this.cardinality = cardinality.clone();
    this.columns = columns.clone();
    for (int col = 0; col < cardinality[COL]; col++) {
      this.columns[col] = columns[col].clone();
    }
  }

  /**
   * Construct a matrix of the given cardinality
   *
   * @param cardinality the int[2] cardinality
   */
  public SparseColumnMatrix(int[] cardinality) {
    this.cardinality = cardinality.clone();
    this.columns = new RandomAccessSparseVector[cardinality[COL]];
    for (int col = 0; col < cardinality[COL]; col++) {
      this.columns[col] = new RandomAccessSparseVector(cardinality[ROW]);
    }
  }

  public int[] size() {
    return cardinality;
  }

  @Override
  public Matrix clone() {
    SparseColumnMatrix clone = (SparseColumnMatrix) super.clone();
    clone.cardinality = cardinality.clone();
    clone.columns = new Vector[columns.length];
    for (int i = 0; i < columns.length; i++) {
      clone.columns[i] = columns[i].clone();
    }
    return clone;
  }

  /**
   * Iterate over columns.
   * @param index of the column number to grab as a vector (shallowly)
   * @return the column vector at that index.
   */
  @Override
  protected Vector slice(int index) {
    return getColumn(index);
  }

  /**
   * Abstracted out for the iterator
   * @return {@link #numCols()} 
   */
  @Override
  public int numSlices() {
    return numCols();
  }

  public double getQuick(int row, int column) {
    return columns[column] == null ? 0.0 : columns[column].getQuick(row);
  }

  public Matrix like() {
    return new SparseColumnMatrix(cardinality);
  }

  public Matrix like(int rows, int columns) {
    int[] c = new int[2];
    c[ROW] = rows;
    c[COL] = columns;
    return new SparseColumnMatrix(c);
  }

  public void setQuick(int row, int column, double value) {
    if (columns[column] == null) {
      columns[column] = new RandomAccessSparseVector(cardinality[ROW]);
    }
    columns[column].setQuick(row, value);
  }

  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[COL] = columns.length;
    for (int col = 0; col < cardinality[COL]; col++) {
      result[ROW] = Math.max(result[ROW], columns[col]
          .getNumNondefaultElements());
    }
    return result;
  }

  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], columns[COL].size());
    }
    if (offset[ROW] + size[ROW] > columns[COL].size()) {
      throw new IndexException(offset[ROW] + size[ROW], columns[COL].size());
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], columns.length);
    }
    if (offset[COL] + size[COL] > columns.length) {
      throw new IndexException(offset[COL] + size[COL], columns.length);
    }
    return new MatrixView(this, offset, size);
  }

  public Matrix assignColumn(int column, Vector other) {
    if (cardinality[ROW] != other.size()) {
      throw new CardinalityException(cardinality[ROW], other.size());
    }
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    columns[column].assign(other);
    return this;
  }

  public Matrix assignRow(int row, Vector other) {
    if (cardinality[COL] != other.size()) {
      throw new CardinalityException(cardinality[COL], other.size());
    }
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    for (int col = 0; col < cardinality[COL]; col++) {
      columns[col].setQuick(row, other.getQuick(col));
    }
    return this;
  }

  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    return columns[column];
  }

  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    return new TransposeViewVector(this, row, false);
  }

}
