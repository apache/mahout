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

import java.util.Arrays;

/** Matrix of doubles implemented using a 2-d array */
public class DenseMatrix extends AbstractMatrix {

  private double[][] values;

  public DenseMatrix() {
  }
  
  /**
   * Construct a matrix from the given values
   * 
   * @param values
   *          a double[][]
   */
  public DenseMatrix(double[][] values) {
    // clone the rows
    this.values = new double[values.length][];
    // be careful, need to clone the columns too
    for (int i = 0; i < values.length; i++) {
      this.values[i] = values[i].clone();
    }
    initSize();
  }
  
  /**
   * Constructs an empty matrix of the given size.
   * @param rows  The number of rows in the result.
   * @param columns The number of columns in the result.
   */
  public DenseMatrix(int rows, int columns) {
    this.values = new double[rows][columns];
    initSize();
  }

  private void initSize() {
    cardinality[ROW] = values.length;
    cardinality[COL] = values[0].length;
  }
  
  @Override
  public Matrix clone() {
    DenseMatrix clone = (DenseMatrix) super.clone();
    clone.values = new double[values.length][];
    for (int i = 0; i < values.length; i++) {
      clone.values[i] = values[i].clone();
    }
    return clone;
  }
  
  public double getQuick(int row, int column) {
    return values[row][column];
  }
  
  public Matrix like() {
    return like(rowSize(), columnSize());
  }
  
  public Matrix like(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }
  
  public void setQuick(int row, int column, double value) {
    values[row][column] = value;
  }
  
  public int[] getNumNondefaultElements() {
    return size();
  }
  
  public Matrix viewPart(int[] offset, int[] size) {
    int rowOffset = offset[ROW];
    int rowsRequested = size[ROW];
    int columnOffset = offset[COL];
    int columnsRequested = size[COL];

    return viewPart(rowOffset, rowsRequested, columnOffset, columnsRequested);
  }

  @Override
  public Matrix viewPart(int rowOffset, int rowsRequested, int columnOffset, int columnsRequested) {
    if (rowOffset < 0) {
      throw new IndexException(rowOffset, rowSize());
    }
    if (rowOffset + rowsRequested > rowSize()) {
      throw new IndexException(rowOffset + rowsRequested, rowSize());
    }
    if (columnOffset < 0) {
      throw new IndexException(columnOffset, columnSize());
    }
    if (columnOffset + columnsRequested > columnSize()) {
      throw new IndexException(columnOffset + columnsRequested, columnSize());
    }
    return new MatrixView(this, new int[]{rowOffset, columnOffset}, new int[]{rowsRequested, columnsRequested});
  }

  @Override
  public Matrix assign(double value) {
    for (int row = 0; row < rowSize(); row++) {
      Arrays.fill(values[row], value);
    }
    return this;
  }
  
  public Matrix assign(DenseMatrix matrix) {
    // make sure the data field has the correct length
    if (matrix.values[0].length != this.values[0].length || matrix.values.length != this.values.length) {
      this.values = new double[matrix.values.length][matrix.values[0].length];
    }
    // now copy the values
    for (int i = 0; i < this.values.length; i++) {
      System.arraycopy(matrix.values[i], 0, this.values[i], 0, this.values[0].length);
    }
    return this;
  }
  
  public Matrix assignColumn(int column, Vector other) {
    if (rowSize() != other.size()) {
      throw new CardinalityException(rowSize(), other.size());
    }
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    for (int row = 0; row < rowSize(); row++) {
      values[row][column] = other.getQuick(row);
    }
    return this;
  }
  
  public Matrix assignRow(int row, Vector other) {
    if (columnSize() != other.size()) {
      throw new CardinalityException(columnSize(), other.size());
    }
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    for (int col = 0; col < columnSize(); col++) {
      values[row][col] = other.getQuick(col);
    }
    return this;
  }
  
  public Vector getColumn(int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return new TransposeViewVector(this, column);
  }
  
  public Vector getRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    return new DenseVector(values[row], true);
  }
  
}
