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

/** Implements subset view of a Matrix */
public class MatrixView extends AbstractMatrix {

  private Matrix matrix;

  // the offset into the Matrix
  private int[] offset;

  // the cardinality of the view
  private int[] cardinality;

  public MatrixView() {
  }

  /**
   * Construct a view of the matrix with given offset and cardinality
   *
   * @param matrix      an underlying Matrix
   * @param offset      the int[2] offset into the underlying matrix
   * @param cardinality the int[2] cardinality of the view
   */
  public MatrixView(Matrix matrix, int[] offset, int[] cardinality) {
    this.matrix = matrix;
    this.offset = offset;
    this.cardinality = cardinality;
  }

  public int[] size() {
    return cardinality;
  }

  @Override
  public Matrix clone() {
    MatrixView clone = (MatrixView) super.clone();
    clone.matrix = matrix.clone();
    clone.offset = offset.clone();
    clone.cardinality = cardinality.clone();
    return clone;
  }

  public double getQuick(int row, int column) {
    return matrix.getQuick(offset[ROW] + row, offset[COL] + column);
  }

  public Matrix like() {
    return matrix.like(cardinality[ROW], cardinality[COL]);
  }

  public Matrix like(int rows, int columns) {

    return matrix.like(rows, columns);
  }

  public void setQuick(int row, int column, double value) {
    matrix.setQuick(offset[ROW] + row, offset[COL] + column, value);
  }

  public int[] getNumNondefaultElements() {
    return cardinality;
  }

  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < ROW) {
      throw new IndexException(offset[ROW], ROW);
    }
    if (offset[ROW] + size[ROW] > cardinality[ROW]) {
      throw new IndexException(offset[ROW] + size[ROW], cardinality[ROW]);
    }
    if (offset[COL] < ROW) {
      throw new IndexException(offset[COL], ROW);
    }
    if (offset[COL] + size[COL] > cardinality[COL]) {
      throw new IndexException(offset[COL] + size[COL], cardinality[COL]);
    }
    int[] origin = offset.clone();
    origin[ROW] += offset[ROW];
    origin[COL] += offset[COL];
    return new MatrixView(matrix, origin, size);
  }

  public Matrix assignColumn(int column, Vector other) {
    if (cardinality[ROW] != other.size()) {
      throw new CardinalityException(cardinality[ROW], other.size());
    }
    for (int row = 0; row < cardinality[ROW]; row++) {
      matrix.setQuick(row + offset[ROW], column + offset[COL], other
          .getQuick(row));
    }
    return this;
  }

  public Matrix assignRow(int row, Vector other) {
    if (cardinality[COL] != other.size()) {
      throw new CardinalityException(cardinality[COL], other.size());
    }
    for (int col = 0; col < cardinality[COL]; col++) {
      matrix
          .setQuick(row + offset[ROW], col + offset[COL], other.getQuick(col));
    }
    return this;
  }

  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    return new VectorView(matrix.getColumn(column + offset[COL]), offset[ROW],
        cardinality[ROW]);
  }

  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    return new VectorView(matrix.getRow(row + offset[ROW]), offset[COL],
        cardinality[COL]);
  }

}
