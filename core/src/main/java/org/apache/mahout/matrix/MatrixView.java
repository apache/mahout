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

/** Implements subset view of a Matrix */
public class MatrixView extends AbstractMatrix {

  private Matrix matrix;

  // the offset into the Matrix
  private int[] offset;

  // the cardinality of the view
  private int[] cardinality;

  public MatrixView() {
    super();
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

  @Override
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

  @Override
  public double getQuick(int row, int column) {
    return matrix.getQuick(offset[ROW] + row, offset[COL] + column);
  }

  @Override
  public Matrix like() {
    return matrix.like(cardinality[ROW], cardinality[COL]);
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
    return cardinality;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (size[ROW] > cardinality[ROW] || size[COL] > cardinality[COL]) {
      throw new CardinalityException();
    }
    if ((offset[ROW] < ROW || offset[ROW] + size[ROW] > cardinality[ROW])
        || (offset[COL] < ROW || offset[COL] + size[COL] > cardinality[COL])) {
      throw new IndexException();
    }
    int[] origin = offset.clone();
    origin[ROW] += offset[ROW];
    origin[COL] += offset[COL];
    return new MatrixView(matrix, origin, size);
  }

  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof MatrixView) {
      return other == this || matrix.haveSharedCells(other);
    } else {
      return other.haveSharedCells(matrix);
    }
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (cardinality[ROW] != other.size()) {
      throw new CardinalityException();
    }
    for (int row = 0; row < cardinality[ROW]; row++) {
      matrix.setQuick(row + offset[ROW], column + offset[COL], other
          .getQuick(row));
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (cardinality[COL] != other.size()) {
      throw new CardinalityException();
    }
    for (int col = 0; col < cardinality[COL]; col++) {
      matrix
          .setQuick(row + offset[ROW], col + offset[COL], other.getQuick(col));
    }
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException();
    }
    return new VectorView(matrix.getColumn(column + offset[COL]), offset[ROW],
        cardinality[ROW]);
  }

  @Override
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException();
    }
    return new VectorView(matrix.getRow(row + offset[ROW]), offset[COL],
        cardinality[COL]);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int[] o = {in.readInt(), in.readInt()};
    this.offset = o;
    int[] c = {in.readInt(), in.readInt()};
    this.cardinality = c;
    this.matrix = readMatrix(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(offset[ROW]);
    out.writeInt(offset[COL]);
    out.writeInt(cardinality[ROW]);
    out.writeInt(cardinality[COL]);
    writeMatrix(out, this.matrix);
  }
}
