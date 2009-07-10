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

package org.apache.mahout.matrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/** Matrix of doubles implemented using a 2-d array */
public class DenseMatrix extends AbstractMatrix {

  private double[][] values;

  public DenseMatrix() {
    super();
  }

  private int columnSize() {
    return values[0].length;
  }

  private int rowSize() {
    return values.length;
  }

  /**
   * Construct a matrix from the given values
   *
   * @param values a double[][]
   */
  public DenseMatrix(double[][] values) {
    // clone the rows
    this.values = new double[values.length][];
    // be careful, need to clone the columns too
    for (int i = 0; i < values.length; i++) {
      this.values[i] = values[i].clone();
    }
  }

  /** Construct an empty matrix of the given size */
  public DenseMatrix(int rows, int columns) {
    this.values = new double[rows][columns];
  }

  @Override
  public int[] size() {
    int[] result = new int[2];
    result[ROW] = rowSize();
    result[COL] = columnSize();
    return result;
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

  @Override
  public double getQuick(int row, int column) {
    return values[row][column];
  }

  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof DenseMatrix) {
      return other == this;
    } else {
      return other.haveSharedCells(this);
    }
  }

  @Override
  public Matrix like() {
    return like(rowSize(), columnSize());
  }

  @Override
  public Matrix like(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    values[row][column] = value;
  }

  @Override
  public int[] getNumNondefaultElements() {
    return size();
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (size[ROW] > rowSize() || size[COL] > columnSize()) {
      throw new CardinalityException();
    }
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > rowSize()
        || offset[COL] < 0 || offset[COL] + size[COL] > columnSize()) {
      throw new IndexException();
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (other.size() != rowSize() || column >= columnSize()) {
      throw new CardinalityException();
    }
    for (int row = 0; row < rowSize(); row++) {
      values[row][column] = other.getQuick(row);
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (row >= rowSize() || other.size() != columnSize()) {
      throw new CardinalityException();
    }
    for (int col = 0; col < columnSize(); col++) {
      values[row][col] = other.getQuick(col);
    }
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException();
    }
    double[] col = new double[rowSize()];
    for (int row = 0; row < rowSize(); row++) {
      col[row] = values[row][column];
    }
    return new DenseVector(col);
  }

  @Override
  public Vector getRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException();
    }
    return new DenseVector(values[row]);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int rows = in.readInt();
    int columns = in.readInt();
    this.values = new double[rows][columns];
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        this.values[row][column] = in.readDouble();
      }
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(rowSize());
    out.writeInt(columnSize());
    for (double[] row : values) {
      for (double value : row) {
        out.writeDouble(value);
      }
    }
  }

}
