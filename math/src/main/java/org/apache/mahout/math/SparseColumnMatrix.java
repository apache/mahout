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

import org.apache.mahout.math.flavor.TraversingStructureEnum;

/**
 * sparse matrix with general element values whose columns are accessible quickly. Implemented as a column array of
 * SparseVectors.
 *
 * @deprecated tons of inconsistences. Use transpose view of SparseRowMatrix for fast column-wise iteration.
 */
public class SparseColumnMatrix extends AbstractMatrix {

  private Vector[] columnVectors;

  /**
   * Construct a matrix of the given cardinality with the given data columns
   *
   * @param columns       a RandomAccessSparseVector[] array of columns
   * @param columnVectors
   */
  public SparseColumnMatrix(int rows, int columns, Vector[] columnVectors) {
    this(rows, columns, columnVectors, false);
  }

  public SparseColumnMatrix(int rows, int columns, Vector[] columnVectors, boolean shallow) {
    super(rows, columns);
    if (shallow) {
      this.columnVectors = columnVectors;
    } else {
      this.columnVectors = columnVectors.clone();
      for (int col = 0; col < columnSize(); col++) {
        this.columnVectors[col] = this.columnVectors[col].clone();
      }
    }
  }

  /**
   * Construct a matrix of the given cardinality
   *
   * @param rows # of rows
   * @param columns # of columns
   */
  public SparseColumnMatrix(int rows, int columns) {
    super(rows, columns);
    this.columnVectors = new RandomAccessSparseVector[columnSize()];
    for (int col = 0; col < columnSize(); col++) {
      this.columnVectors[col] = new RandomAccessSparseVector(rowSize());
    }
  }

  @Override
  public Matrix clone() {
    SparseColumnMatrix clone = (SparseColumnMatrix) super.clone();
    clone.columnVectors = new Vector[columnVectors.length];
    for (int i = 0; i < columnVectors.length; i++) {
      clone.columnVectors[i] = columnVectors[i].clone();
    }
    return clone;
  }

  /**
   * Abstracted out for the iterator
   *
   * @return {@link #numCols()}
   */
  @Override
  public int numSlices() {
    return numCols();
  }

  @Override
  public double getQuick(int row, int column) {
    return columnVectors[column] == null ? 0.0 : columnVectors[column].getQuick(row);
  }

  @Override
  public Matrix like() {
    return new SparseColumnMatrix(rowSize(), columnSize());
  }

  @Override
  public Matrix like(int rows, int columns) {
    return new SparseColumnMatrix(rows, columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    if (columnVectors[column] == null) {
      columnVectors[column] = new RandomAccessSparseVector(rowSize());
    }
    columnVectors[column].setQuick(row, value);
  }

  @Override
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[COL] = columnVectors.length;
    for (int col = 0; col < columnSize(); col++) {
      result[ROW] = Math.max(result[ROW], columnVectors[col]
        .getNumNondefaultElements());
    }
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], columnVectors[COL].size());
    }
    if (offset[ROW] + size[ROW] > columnVectors[COL].size()) {
      throw new IndexException(offset[ROW] + size[ROW], columnVectors[COL].size());
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], columnVectors.length);
    }
    if (offset[COL] + size[COL] > columnVectors.length) {
      throw new IndexException(offset[COL] + size[COL], columnVectors.length);
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (rowSize() != other.size()) {
      throw new CardinalityException(rowSize(), other.size());
    }
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    columnVectors[column].assign(other);
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (columnSize() != other.size()) {
      throw new CardinalityException(columnSize(), other.size());
    }
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    for (int col = 0; col < columnSize(); col++) {
      columnVectors[col].setQuick(row, other.getQuick(col));
    }
    return this;
  }

  @Override
  public Vector viewColumn(int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return columnVectors[column];
  }

  @Override
  public Matrix transpose() {
    SparseRowMatrix srm = new SparseRowMatrix(columns, rows);
    for (int i = 0; i < columns; i++) {
      Vector col = columnVectors[i];
      if (col.getNumNonZeroElements() > 0)
        // this should already be optimized
        srm.assignRow(i, col);
    }
    return srm;
  }

  @Override
  public String toString() {
    int row = 0;
    int maxRowsToDisplay = 10;
    int maxColsToDisplay = 20;
    int colsToDisplay = maxColsToDisplay;

    if(maxColsToDisplay > columnSize()){
      colsToDisplay = columnSize();
    }

    StringBuilder s = new StringBuilder("{\n");
    for (MatrixSlice next : this.transpose()) {
      if (row < maxRowsToDisplay) {
        s.append(" ")
          .append(next.index())
          .append(" =>\t")
          .append(new VectorView(next.vector(), 0, colsToDisplay))
          .append('\n');
        row++;
      }
    }

    String returnString = s.toString();
    if (maxColsToDisplay <= columnSize()) {
      returnString = returnString.replace("}", " ... }");
    }

    if (maxRowsToDisplay <= rowSize()) {
      return returnString + "... }";
    } else {
      return returnString + "}";
    }
  }

}
