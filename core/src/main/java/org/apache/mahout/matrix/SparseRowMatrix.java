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

/**
 * sparse matrix with general element values whose rows are accessible quickly. Implemented as a row array of
 * SparseVectors.
 */
public class SparseRowMatrix extends AbstractMatrix {

  private int[] cardinality;

  private Vector[] rows;

  public SparseRowMatrix() {
    super();
  }

  /**
   * Construct a matrix of the given cardinality with the given rows
   *
   * @param cardinality the int[2] cardinality desired
   * @param rows        a SparseVector[] array of rows
   */
  public SparseRowMatrix(int[] cardinality, SparseVector[] rows) {
    this.cardinality = cardinality.clone();
    this.rows = rows.clone();
    for (int row = 0; row < cardinality[ROW]; row++) {
      this.rows[row] = rows[row].clone();
    }
  }

  /**
   * Construct a matrix of the given cardinality
   *
   * @param cardinality the int[2] cardinality desired
   */
  public SparseRowMatrix(int[] cardinality) {
    this.cardinality = cardinality.clone();
    this.rows = new SparseVector[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++) {
      this.rows[row] = new SparseVector(cardinality[COL]);
    }
  }

  @Override
  public int[] size() {
    return cardinality;
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

  @Override
  public double getQuick(int row, int column) {
    if (rows[row] == null) {
      return 0.0;
    } else {
      return rows[row].getQuick(column);
    }
  }

  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof SparseRowMatrix) {
      return other == this;
    }
    return other.haveSharedCells(this);
  }

  @Override
  public Matrix like() {
    return new SparseRowMatrix(cardinality);
  }

  @Override
  public Matrix like(int rows, int columns) {
    int[] c = new int[2];
    c[ROW] = rows;
    c[COL] = columns;
    return new SparseRowMatrix(c);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    rows[row].setQuick(column, value);
  }

  @Override
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rows.length;
    for (int row = 0; row < cardinality[ROW]; row++) {
      result[COL] = Math.max(result[COL], rows[row].getNumNondefaultElements());
    }
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (size[ROW] > rows.length || size[COL] > rows[ROW].size()) {
      throw new CardinalityException();
    }
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > rows.length
        || offset[COL] < 0 || offset[COL] + size[COL] > rows[ROW].size()) {
      throw new IndexException();
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (other.size() != cardinality[ROW] || column >= cardinality[COL]) {
      throw new CardinalityException();
    }
    for (int row = 0; row < cardinality[ROW]; row++) {
      rows[row].setQuick(column, other.getQuick(row));
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (row >= cardinality[ROW] || other.size() != cardinality[COL]) {
      throw new CardinalityException();
    }
    rows[row].assign(other);
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException();
    }
    double[] d = new double[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++) {
      d[row] = getQuick(row, column);
    }
    return new DenseVector(d);
  }

  @Override
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException();
    }
    return rows[row];
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int[] card = {in.readInt(), in.readInt()};
    this.cardinality = card;
    int rowsize = in.readInt();
    this.rows = new Vector[rowsize];
    for (int row = 0; row < rowsize; row++) {
      rows[row] = AbstractVector.readVector(in);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(cardinality[ROW]);
    out.writeInt(cardinality[COL]);
    out.writeInt(rows.length);
    for (Vector row : rows) {
      AbstractVector.writeVector(out, row);
    }
  }

}
