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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

/**
 * sparse matrix with general element values whose columns are accessible
 * quickly. Implemented as a column array of SparseVectors.
 */
public class SparseColumnMatrix extends AbstractMatrix {

  private final int[] cardinality;

  private final Vector[] columns;

  /**
   * Construct a matrix of the given cardinality with the given data columns
   * 
   * @param cardinality
   *            the int[2] cardinality
   * @param columns
   *            a SparseVector[] array of columns
   */
  public SparseColumnMatrix(int[] cardinality, SparseVector[] columns) {
    super();
    this.cardinality = cardinality.clone();
    this.columns = columns.clone();
    for (int col = 0; col < cardinality[COL]; col++)
      this.columns[col] = columns[col].copy();
  }

  /**
   * Construct a matrix of the given cardinality
   * 
   * @param cardinality
   *            the int[2] cardinality
   */
  public SparseColumnMatrix(int[] cardinality) {
    super();
    this.cardinality = cardinality.clone();
    this.columns = new SparseVector[cardinality[COL]];
    for (int col = 0; col < cardinality[COL]; col++)
      this.columns[col] = new SparseVector(cardinality[ROW]);
  }

  @Override
  public WritableComparable asWritableComparable() {
    String out = asFormatString();
    return new Text(out);
  }

  @Override
  public String asFormatString() {
    StringBuilder out = new StringBuilder();
    out.append("[[, ");
    for (int row = 0; row < columns[ROW].size(); row++) {
      for (int col = 0; col < columns.length; col++)
        out.append(getQuick(row, col)).append(", ");
      out.append("], ");
    }
    out.append("] ");
    return out.toString();
  }

  @Override
  public int[] cardinality() {
    return cardinality;
  }

  @Override
  public Matrix copy() {
    SparseColumnMatrix copy = new SparseColumnMatrix(cardinality);
    for (int col = 0; col < cardinality[COL]; col++)
      copy.columns[col] = columns[col].copy();
    return copy;
  }

  @Override
  public double getQuick(int row, int column) {
    if (columns[column] == null)
      return 0.0;
    else
      return columns[column].getQuick(row);
  }

  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof SparseColumnMatrix)
      return other == this;
    return other.haveSharedCells(this);
  }

  @Override
  public Matrix like() {
    return new SparseColumnMatrix(cardinality);
  }


  @Override
  public Matrix like(int rows, int columns) {
    int[] c = new int[2];
    c[ROW] = rows;
    c[COL] = columns;
    return new SparseColumnMatrix(c);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    if (columns[column] == null)
      columns[column] = new SparseVector(cardinality[ROW]);
    columns[column].setQuick(row, value);
  }

  @Override
  public int[] size() {
    int[] result = new int[2];
    result[COL] = columns.length;
    for (int col = 0; col < cardinality[COL]; col++)
      result[ROW] = Math.max(result[ROW], columns[col].size());
    return result;
  }

  @Override
  public double[][] toArray() {
    double[][] result = new double[cardinality[ROW]][cardinality[COL]];
    for (int row = 0; row < cardinality[ROW]; row++)
      for (int col = 0; col < cardinality[COL]; col++)
        result[row][col] = getQuick(row, col);
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (size[COL] > columns.length || size[ROW] > columns[COL].cardinality())
      throw new CardinalityException();
    if (offset[COL] < 0 || offset[COL] + size[COL] > columns.length
        || offset[ROW] < 0
        || offset[ROW] + size[ROW] > columns[COL].cardinality())
      throw new IndexException();
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (other.cardinality() != cardinality[ROW] || column >= cardinality[COL])
      throw new CardinalityException();
    columns[column].assign(other);
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (row >= cardinality[ROW] || other.cardinality() != cardinality[COL])
      throw new CardinalityException();
    for (int col = 0; col < cardinality[COL]; col++)
      columns[col].setQuick(row, other.getQuick(col));
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL])
      throw new IndexException();
    return columns[column];
  }

  @Override
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW])
      throw new IndexException();
    double[] d = new double[cardinality[COL]];
    for (int col = 0; col < cardinality[COL]; col++)
      d[col] = getQuick(row, col);
    return new DenseVector(d);
  }

}
