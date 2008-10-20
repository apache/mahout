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
 * Implements subset view of a Matrix
 */
public class MatrixView extends AbstractMatrix {

  private final Matrix matrix;

  // the offset into the Matrix
  private final int[] offset;

  // the cardinality of the view
  private final int[] cardinality;

  /**
   * Construct a view of the matrix with given offset and cardinality
   * 
   * @param matrix
   *            an underlying Matrix
   * @param offset
   *            the int[2] offset into the underlying matrix
   * @param cardinality
   *            the int[2] cardinality of the view
   */
  public MatrixView(Matrix matrix, int[] offset, int[] cardinality) {
    super();
    this.matrix = matrix;
    this.offset = offset;
    this.cardinality = cardinality;
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
    out.append("[, ");
    for (int row = 0; row < cardinality[ROW]; row++) {
      out.append("[, ");
      for (int col = 0; col < cardinality[COL]; col++)
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
    return new MatrixView(matrix.copy(), offset, cardinality);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getQuick(int, int)
   */
  @Override
  public double getQuick(int row, int column) {
    return matrix.getQuick(offset[ROW] + row, offset[COL] + column);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like()
   */
  @Override
  public Matrix like() {
    return matrix.like(cardinality[ROW], cardinality[COL]);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#like(int, int)
   */
  @Override
  public Matrix like(int rows, int columns) {

    return matrix.like(rows, columns);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#setQuick(int, int, double)
   */
  @Override
  public void setQuick(int row, int column, double value) {
    matrix.setQuick(offset[ROW] + row, offset[COL] + column, value);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#size()
   */
  @Override
  public int[] size() {
    return cardinality;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#toArray()
   */
  @Override
  public double[][] toArray() {
    double[][] result = new double[cardinality[ROW]][cardinality[COL]];
    for (int row = ROW; row < cardinality[ROW]; row++)
      for (int col = ROW; col < cardinality[COL]; col++)
        result[row][col] = matrix
            .getQuick(offset[ROW] + row, offset[COL] + col);
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) throws CardinalityException,
      IndexException {
    if (size[ROW] > cardinality[ROW] || size[COL] > cardinality[COL])
      throw new CardinalityException();
    if ((offset[ROW] < ROW || offset[ROW] + size[ROW] > cardinality[ROW])
        || (offset[COL] < ROW || offset[COL] + size[COL] > cardinality[COL]))
      throw new IndexException();
    int[] origin = offset.clone();
    origin[ROW] += offset[ROW];
    origin[COL] += offset[COL];
    return new MatrixView(matrix, origin, size);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#haveSharedCells(org.apache.mahout.matrix.Matrix)
   */
  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof MatrixView)
      return other == this || matrix.haveSharedCells(other);
    else
      return other.haveSharedCells(matrix);
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
    if (cardinality[ROW] != other.cardinality())
      throw new CardinalityException();
    for (int row = 0; row < cardinality[ROW]; row++)
      matrix.setQuick(row + offset[ROW], column + offset[COL], other
          .getQuick(row));
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
    if (cardinality[COL] != other.cardinality())
      throw new CardinalityException();
    for (int col = 0; col < cardinality[COL]; col++)
      matrix
          .setQuick(row + offset[ROW], col + offset[COL], other.getQuick(col));
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
    return new VectorView(matrix.getColumn(column + offset[COL]), offset[ROW],
        cardinality[ROW]);
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
    return new VectorView(matrix.getRow(row + offset[ROW]), offset[COL],
        cardinality[COL]);
  }
}
