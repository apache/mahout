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

  /**
   * Construct a view of the matrix with given offset and cardinality
   *
   * @param matrix      an underlying Matrix
   * @param offset      the int[2] offset into the underlying matrix
   * @param size        the int[2] size of the view
   */
  public MatrixView(Matrix matrix, int[] offset, int[] size) {
    super(size[ROW], size[COL]);
    int rowOffset = offset[ROW];
    if (rowOffset < 0) {
      throw new IndexException(rowOffset, rowSize());
    }

    int rowsRequested = size[ROW];
    if (rowOffset + rowsRequested > matrix.rowSize()) {
      throw new IndexException(rowOffset + rowsRequested, matrix.rowSize());
    }

    int columnOffset = offset[COL];
    if (columnOffset < 0) {
      throw new IndexException(columnOffset, columnSize());
    }

    int columnsRequested = size[COL];
    if (columnOffset + columnsRequested > matrix.columnSize()) {
      throw new IndexException(columnOffset + columnsRequested, matrix.columnSize());
    }
    this.matrix = matrix;
    this.offset = offset;
  }

  @Override
  public Matrix clone() {
    MatrixView clone = (MatrixView) super.clone();
    clone.matrix = matrix.clone();
    clone.offset = offset.clone();
    return clone;
  }

  @Override
  public double getQuick(int row, int column) {
    return matrix.getQuick(offset[ROW] + row, offset[COL] + column);
  }

  @Override
  public Matrix like() {
    return matrix.like(rowSize(), columnSize());
  }

  @Override
  public Matrix like(int rows, int columns) {
    return matrix.like(rows, columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    matrix.setQuick(offset[ROW] + row, offset[COL] + column, value);
  }

  @Override
  public int[] getNumNondefaultElements() {
    return new int[]{rowSize(), columnSize()};

  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], 0);
    }
    if (offset[ROW] + size[ROW] > rowSize()) {
      throw new IndexException(offset[ROW] + size[ROW], rowSize());
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], 0);
    }
    if (offset[COL] + size[COL] > columnSize()) {
      throw new IndexException(offset[COL] + size[COL], columnSize());
    }
    int[] origin = this.offset.clone();
    origin[ROW] += offset[ROW];
    origin[COL] += offset[COL];
    return new MatrixView(matrix, origin, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (rowSize() != other.size()) {
      throw new CardinalityException(rowSize(), other.size());
    }
    for (int row = 0; row < rowSize(); row++) {
      matrix.setQuick(row + offset[ROW], column + offset[COL], other
          .getQuick(row));
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (columnSize() != other.size()) {
      throw new CardinalityException(columnSize(), other.size());
    }
    for (int col = 0; col < columnSize(); col++) {
      matrix
          .setQuick(row + offset[ROW], col + offset[COL], other.getQuick(col));
    }
    return this;
  }

  @Override
  public Vector viewColumn(int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return new VectorView(matrix.viewColumn(column + offset[COL]), offset[ROW], rowSize());
  }

  @Override
  public Vector viewRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    return new VectorView(matrix.viewRow(row + offset[ROW]), offset[COL], columnSize());
  }

}
