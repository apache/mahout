/*
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

public class DiagonalMatrix extends AbstractMatrix implements MatrixTimesOps {
  private final Vector diagonal;

  public DiagonalMatrix(Vector values) {
    super(values.size(), values.size());
    this.diagonal = values;
  }

  public DiagonalMatrix(Matrix values) {
    this(values.viewDiagonal());
  }

  public DiagonalMatrix(double value, int size) {
    this(new ConstantVector(value, size));
  }

  public DiagonalMatrix(double[] values) {
    super(values.length, values.length);
    this.diagonal = new DenseVector(values);
  }

  public static DiagonalMatrix identity(int size) {
    return new DiagonalMatrix(1, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    throw new UnsupportedOperationException("Can't assign a column to a diagonal matrix");
  }

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  @Override
  public Matrix assignRow(int row, Vector other) {
    throw new UnsupportedOperationException("Can't assign a row to a diagonal matrix");
  }

  /**
   * Provides a view of the diagonal of a matrix.
   */
  @Override
  public Vector viewDiagonal() {
    return this.diagonal;
  }

  /**
   * Return the value at the given location, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int row, int column) {
    if (row == column) {
      return diagonal.get(row);
    } else {
      return 0;
    }
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    return new SparseRowMatrix(rowSize(), columnSize());
  }

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified
   * size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  @Override
  public Matrix like(int rows, int columns) {
    return new SparseRowMatrix(rows, columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    if (row == column) {
      diagonal.set(row, value);
    } else {
      throw new UnsupportedOperationException("Can't set off-diagonal element");
    }
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  @Override
  public int[] getNumNondefaultElements() {
    throw new UnsupportedOperationException("Don't understand how to implement this");
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the
   *                              receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix times(Matrix other) {
    return timesRight(other);
  }

  @Override
  public Matrix timesRight(Matrix that) {
    if (that.numRows() != diagonal.size())
      throw new IllegalArgumentException("Incompatible number of rows in the right operand of matrix multiplication.");
    Matrix m = that.like();
    for (int row = 0; row < diagonal.size(); row++)
      m.assignRow(row, that.viewRow(row).times(diagonal.getQuick(row)));
    return m;
  }

  @Override
  public Matrix timesLeft(Matrix that) {
    if (that.numCols() != diagonal.size())
      throw new IllegalArgumentException(
          "Incompatible number of rows in the left operand of matrix-matrix multiplication.");
    Matrix m = that.like();
    for (int col = 0; col < diagonal.size(); col++)
      m.assignColumn(col, that.viewColumn(col).times(diagonal.getQuick(col)));
    return m;
  }
}
