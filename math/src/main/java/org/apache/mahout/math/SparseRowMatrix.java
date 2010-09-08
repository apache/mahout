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
 * sparse matrix with general element values whose rows are accessible quickly. Implemented as a row array of
 * either SequentialAccessSparseVectors or RandomAccessSparseVectors.
 */
public class SparseRowMatrix extends AbstractMatrix {
  private Vector[] rows;

  private boolean randomAccessRows;

  public SparseRowMatrix() {
  }

  /**
   * Construct a matrix of the given cardinality with the given rows
   *
   * @param cardinality the int[2] cardinality desired
   * @param rows        a Vector[] array of rows
   */
  public SparseRowMatrix(int[] cardinality, Vector[] rows) {
    this(cardinality, rows, false, rows instanceof RandomAccessSparseVector[]);
  }

  public SparseRowMatrix(int[] cardinality, boolean randomAccess) {
    this(cardinality, randomAccess
                    ? new RandomAccessSparseVector[cardinality[ROW]]
                    : new SequentialAccessSparseVector[cardinality[ROW]],
        true,
        randomAccess);
  }
  
  public SparseRowMatrix(int[] cardinality, Vector[] rows, boolean shallowCopy, boolean randomAccess) {
    this.cardinality = cardinality.clone();
    this.randomAccessRows = randomAccess;
    this.rows = rows.clone();
    for (int row = 0; row < cardinality[ROW]; row++) {
      if (rows[row] == null) {
        rows[row] = randomAccess
                  ? new RandomAccessSparseVector(numCols(), 10)
                  : new SequentialAccessSparseVector(numCols(), 10);
      }
      this.rows[row] = shallowCopy ? rows[row] : rows[row].clone();
    }
  }

  /**
   * Construct a matrix of the given cardinality, with rows defaulting to RandomAccessSparseVector implementation
   *
   * @param cardinality the int[2] cardinality desired
   */
  public SparseRowMatrix(int[] cardinality) {
    this(cardinality, true);
  }

  @Override
  public Matrix clone() {
    SparseRowMatrix clone = (SparseRowMatrix) super.clone();
    clone.cardinality = cardinality.clone();
    clone.rows = new Vector[rows.length];
    for (int i = 0; i < rows.length; i++) {
      clone.rows[i] = rows[i].clone();
    }
    return clone;
  }

  public double getQuick(int row, int column) {
    return rows[row] == null ? 0.0 : rows[row].getQuick(column);
  }

  public Matrix like() {
    return new SparseRowMatrix(cardinality, randomAccessRows);
  }

  public Matrix like(int rows, int columns) {
    int[] c = new int[2];
    c[ROW] = rows;
    c[COL] = columns;
    return new SparseRowMatrix(c, randomAccessRows);
  }

  public void setQuick(int row, int column, double value) {
    rows[row].setQuick(column, value);
  }

  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rows.length;
    for (int row = 0; row < cardinality[ROW]; row++) {
      result[COL] = Math.max(result[COL], rows[row].getNumNondefaultElements());
    }
    return result;
  }

  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], rows.length);
    }
    if (offset[ROW] + size[ROW] > rows.length) {
      throw new IndexException(offset[ROW] + size[ROW], rows.length);
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], rows[ROW].size());
    }
    if (offset[COL] + size[COL] > rows[ROW].size()) {
      throw new IndexException(offset[COL] + size[COL], rows[ROW].size());
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
    for (int row = 0; row < cardinality[ROW]; row++) {
      rows[row].setQuick(column, other.getQuick(row));
    }
    return this;
  }

  public Matrix assignRow(int row, Vector other) {
    if (cardinality[COL] != other.size()) {
      throw new CardinalityException(cardinality[COL], other.size());
    }
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    rows[row].assign(other);
    return this;
  }

  /**
   *
   * @param column an int column index
   * @return a shallow view of the column
   */
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    return new TransposeViewVector(this, column) {
      @Override
      protected Vector newVector(int cardinality) {
        return randomAccessRows
             ? new RandomAccessSparseVector(cardinality, 10)
             : new SequentialAccessSparseVector(cardinality, 10);
      }
    };
  }

  /**
   *
   * @param row an int row index
   * @return a shallow view of the Vector at specified row (ie you may mutate the original matrix using this row)
   */
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    return rows[row];
  }

}
