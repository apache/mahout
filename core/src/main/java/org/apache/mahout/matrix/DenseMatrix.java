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
 * Matrix of doubles implemented using a 2-d array
 * 
 */
public class DenseMatrix extends AbstractMatrix {

  private double[][] values;

  private int columnSize() {
    return values[0].length;
  }

  private int rowSize() {
    return values.length;
  }

  /**
   * Construct a matrix from the given values
   * 
   * @param values
   *            a double[][]
   */
  public DenseMatrix(double[][] values) {
    super();
    // clone the rows
    this.values = values.clone();
    // be careful, need to clone the columns too
    for (int i = 0; i < values.length; i++)
      this.values[i] = this.values[i].clone();
  }

  /**
   * Construct an empty matrix of the given size
   * 
   * @param rows
   * @param columns
   */
  public DenseMatrix(int rows, int columns) {
    super();
    this.values = new double[rows][columns];
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#asFormatString()
   */
  @Override
  public WritableComparable asWritableComparable() {
    return new Text(asFormatString());
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
    for (int row = 0; row < rowSize(); row++) {
      for (int col = 0; col < values[row].length; col++)
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
    int[] result = new int[2];
    result[ROW] = rowSize();
    result[COL] = columnSize();
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#copy()
   */
  @Override
  public Matrix copy() {
    return new DenseMatrix(values);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getQuick(int, int)
   */
  @Override
  public double getQuick(int row, int column) {
    return values[row][column];
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#haveSharedCells(org.apache.mahout.matrix.Matrix)
   */
  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof DenseMatrix)
      return other == this;
    else
      return other.haveSharedCells(this);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like()
   */
  @Override
  public Matrix like() {
    return like(rowSize(), columnSize());
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like(int, int)
   */
  @Override
  public Matrix like(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#setQuick(int, int, double)
   */
  @Override
  public void setQuick(int row, int column, double value) {
    values[row][column] = value;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#size()
   */
  @Override
  public int[] size() {
    return cardinality();
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#toArray()
   */
  @Override
  public double[][] toArray() {
    DenseMatrix result = new DenseMatrix(values);
    return result.values;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#viewPart(int[], int[])
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) throws CardinalityException,
      IndexException {
    if (size[ROW] > rowSize() || size[COL] > columnSize())
      throw new CardinalityException();
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > rowSize()
        || offset[COL] < 0 || offset[COL] + size[COL] > columnSize())
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
    if (other.cardinality() != rowSize() || column >= columnSize())
      throw new CardinalityException();
    for (int row = 0; row < rowSize(); row++)
      values[row][column] = other.getQuick(row);
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
    if (row >= rowSize() || other.cardinality() != columnSize())
      throw new CardinalityException();
    values[row] = other.toArray();
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getColumn(int)
   */
  @Override
  public Vector getColumn(int column) throws IndexException {
    if (column < 0 || column >= columnSize())
      throw new IndexException();
    double[] col = new double[rowSize()];
    for (int row = 0; row < rowSize(); row++)
      col[row] = values[row][column];
    return new DenseVector(col);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getRow(int)
   */
  @Override
  public Vector getRow(int row) throws IndexException {
    if (row < 0 || row >= rowSize())
      throw new IndexException();
    return new DenseVector(values[row]);
  }

}
