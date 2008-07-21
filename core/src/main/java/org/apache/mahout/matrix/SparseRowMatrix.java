/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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
 * sparse matrix with general element values whose rows are accessible quickly.
 * Implemented as a row array of SparseVectors.
 */
public class SparseRowMatrix extends AbstractMatrix {

  private final int[] cardinality;

  private final Vector[] rows;

  /**
   * Construct a matrix of the given cardinality with the given rows
   * 
   * @param cardinality
   *            the int[2] cardinality desired
   * @param rows
   *            a SparseVector[] array of rows
   */
  public SparseRowMatrix(int[] cardinality, SparseVector[] rows) {
    this.cardinality = cardinality.clone();
    this.rows = rows.clone();
    for (int row = 0; row < cardinality[ROW]; row++)
      this.rows[row] = rows[row].copy();
  }

  /**
   * Construct a matrix of the given cardinality
   * 
   * @param cardinality
   *            the int[2] cardinality desired
   */
  public SparseRowMatrix(int[] cardinality) {
    super();
    this.cardinality = cardinality.clone();
    this.rows = new SparseVector[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++)
      this.rows[row] = new SparseVector(cardinality[COL]);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#asFormatString()
   */
  @Override
  public WritableComparable asWritableComparable() {
    String out = asFormatString();
    return new Text(out);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#asFormatString()
   */
  @Override
  public String asFormatString() {
    StringBuilder out = new StringBuilder();
    out.append("[[, ");
    for (int row = 0; row < rows.length; row++) {
      for (int col = 0; col < rows[ROW].size(); col++)
        out.append(getQuick(row, col)).append(", ");
      out.append("], ");
    }
    out.append("] ");
    return out.toString();
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#cardinality()
   */
  @Override
  public int[] cardinality() {
    return cardinality;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#copy()
   */
  @Override
  public Matrix copy() {
    SparseRowMatrix copy = new SparseRowMatrix(cardinality);
    for (int row = 0; row < cardinality[ROW]; row++)
      copy.rows[row] = rows[row].copy();
    return copy;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getQuick(int, int)
   */
  @Override
  public double getQuick(int row, int column) {
    if (rows[row] == null)
      return 0.0;
    else
      return rows[row].getQuick(column);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#haveSharedCells(org.apache.mahout.matrix.Matrix)
   */
  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof SparseRowMatrix)
      return other == this;
    return other.haveSharedCells(this);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like()
   */
  @Override
  public Matrix like() {
    return new SparseRowMatrix(cardinality);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like(int, int)
   */
  @Override
  public Matrix like(int rows, int columns) {
    int[] c = new int[2];
    c[ROW] = rows;
    c[COL] = columns;
    return new SparseRowMatrix(c);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#setQuick(int, int, double)
   */
  @Override
  public void setQuick(int row, int column, double value) {
    rows[row].setQuick(column, value);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#size()
   */
  @Override
  public int[] size() {
    int[] result = new int[2];
    result[ROW] = rows.length;
    for (int row = 0; row < cardinality[ROW]; row++)
      result[COL] = Math.max(result[COL], rows[row].size());
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#toArray()
   */
  @Override
  public double[][] toArray() {
    double[][] result = new double[cardinality[ROW]][cardinality[COL]];
    for (int row = 0; row < cardinality[ROW]; row++)
      for (int col = 0; col < cardinality[COL]; col++)
        result[row][col] = getQuick(row, col);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#viewPart(int[], int[])
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) throws CardinalityException,
      IndexException {
    if (size[ROW] > rows.length || size[COL] > rows[ROW].cardinality())
      throw new CardinalityException();
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > rows.length
        || offset[COL] < 0 || offset[COL] + size[COL] > rows[ROW].cardinality())
      throw new IndexException();
    return new MatrixView(this, offset, size);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#assignColumn(int,
   *      org.apache.mahout.vector.Vector)
   */
  @Override
  public Matrix assignColumn(int column, Vector other)
      throws CardinalityException {
    if (other.cardinality() != cardinality[ROW] || column >= cardinality[COL])
      throw new CardinalityException();
    for (int row = 0; row < cardinality[ROW]; row++)
      rows[row].setQuick(column, other.getQuick(row));
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#assignRow(int,
   *      org.apache.mahout.vector.Vector)
   */
  @Override
  public Matrix assignRow(int row, Vector other) throws CardinalityException {
    if (row >= cardinality[ROW] || other.cardinality() != cardinality[COL])
      throw new CardinalityException();
    rows[row].assign(other);
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getColumn(int)
   */
  @Override
  public Vector getColumn(int column) throws IndexException {
    if (column < 0 || column >= cardinality[COL])
      throw new IndexException();
    double[] d = new double[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++)
      d[row] = getQuick(row, column);
    return new DenseVector(d);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getRow(int)
   */
  @Override
  public Vector getRow(int row) throws IndexException {
    if (row < 0 || row >= cardinality[ROW])
      throw new IndexException();
    return rows[row];
  }

}
