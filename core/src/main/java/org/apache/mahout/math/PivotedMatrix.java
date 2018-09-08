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

import com.google.common.base.Preconditions;

/**
 * Matrix that allows transparent row and column permutation.
 */
public class PivotedMatrix extends AbstractMatrix {

  private Matrix base;
  private int[] rowPivot;
  private int[] rowUnpivot;
  private int[] columnPivot;
  private int[] columnUnpivot;

  public PivotedMatrix(Matrix base, int[] pivot) {
    this(base, pivot, java.util.Arrays.copyOf(pivot, pivot.length));
  }
  public PivotedMatrix(Matrix base, int[] rowPivot, int[] columnPivot) {
    super(base.rowSize(), base.columnSize());

    this.base = base;
    this.rowPivot = rowPivot;
    rowUnpivot = invert(rowPivot);

    this.columnPivot = columnPivot;
    columnUnpivot = invert(columnPivot);
  }

  public PivotedMatrix(Matrix base) {
    this(base, identityPivot(base.rowSize()),identityPivot(base.columnSize()));
  }

  /**
   * Swaps indexes i and j.  This does both row and column permutation.
   *
   * @param i First index to swap.
   * @param j Second index to swap.
   */
  public void swap(int i, int j) {
    swapRows(i, j);
    swapColumns(i, j);
  }

  /**
   * Swaps indexes i and j.  This does just row permutation.
   *
   * @param i First index to swap.
   * @param j Second index to swap.
   */
  public void swapRows(int i, int j) {
    swap(rowPivot, rowUnpivot, i, j);
  }


  /**
   * Swaps indexes i and j.  This does just row permutation.
   *
   * @param i First index to swap.
   * @param j Second index to swap.
   */
  public void swapColumns(int i, int j) {
    swap(columnPivot, columnUnpivot, i, j);
  }

  private static void swap(int[] pivot, int[] unpivot, int i, int j) {
    Preconditions.checkPositionIndex(i, pivot.length);
    Preconditions.checkPositionIndex(j, pivot.length);
    if (i != j) {
      int tmp = pivot[i];
      pivot[i] = pivot[j];
      pivot[j] = tmp;

      unpivot[pivot[i]] = i;
      unpivot[pivot[j]] = j;
    }
  }

  /**
   * Assign the other vector values to the column of the receiver
   *
   * @param column the int row to assign
   * @param other  a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignColumn(int column, Vector other) {
    // note the reversed pivoting for other
    return base.assignColumn(columnPivot[column], new PermutedVectorView(other, rowUnpivot, rowPivot));
  }

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws org.apache.mahout.math.CardinalityException
   *          if the cardinalities differ
   */
  @Override
  public Matrix assignRow(int row, Vector other) {
    // note the reversed pivoting for other
    return base.assignRow(rowPivot[row], new PermutedVectorView(other, columnUnpivot, columnPivot));
  }

  /**
   * Return the column at the given index
   *
   * @param column an int column index
   * @return a Vector at the index
   * @throws org.apache.mahout.math.IndexException
   *          if the index is out of bounds
   */
  @Override
  public Vector viewColumn(int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return new PermutedVectorView(base.viewColumn(columnPivot[column]), rowPivot, rowUnpivot);
  }

  /**
   * Return the row at the given index
   *
   * @param row an int row index
   * @return a Vector at the index
   * @throws org.apache.mahout.math.IndexException
   *          if the index is out of bounds
   */
  @Override
  public Vector viewRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    return new PermutedVectorView(base.viewRow(rowPivot[row]), columnPivot, columnUnpivot);
  }

  /**
   * Return the value at the given indexes, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  @Override
  public double getQuick(int row, int column) {
    return base.getQuick(rowPivot[row], columnPivot[column]);
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  @Override
  public Matrix like() {
    return new PivotedMatrix(base.like());
  }


  @Override
  public Matrix clone() {
    PivotedMatrix clone = (PivotedMatrix) super.clone();

    base = base.clone();
    rowPivot = rowPivot.clone();
    rowUnpivot = rowUnpivot.clone();
    columnPivot = columnPivot.clone();
    columnUnpivot = columnUnpivot.clone();

    return clone;
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
    return new PivotedMatrix(base.like(rows, columns));
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   */
  @Override
  public void setQuick(int row, int column, double value) {
    base.setQuick(rowPivot[row], columnPivot[column], value);
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  @Override
  public int[] getNumNondefaultElements() {
    return base.getNumNondefaultElements();
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws org.apache.mahout.math.CardinalityException
   *          if the length is greater than the cardinality of the receiver
   * @throws org.apache.mahout.math.IndexException
   *          if the offset is negative or the offset+length is outside of the receiver
   */
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    return new MatrixView(this, offset, size);
  }

  public int rowUnpivot(int k) {
    return rowUnpivot[k];
  }

  public int columnUnpivot(int k) {
    return columnUnpivot[k];
  }

  public int[] getRowPivot() {
    return rowPivot;
  }

  public int[] getInverseRowPivot() {
    return rowUnpivot;
  }

  public int[] getColumnPivot() {
    return columnPivot;
  }

  public int[] getInverseColumnPivot() {
    return columnUnpivot;
  }

  public Matrix getBase() {
    return base;
  }

  private static int[] identityPivot(int n) {
    int[] pivot = new int[n];
    for (int i = 0; i < n; i++) {
      pivot[i] = i;
    }
    return pivot;
  }

  private static int[] invert(int[] pivot) {
    int[] x = new int[pivot.length];
    for (int i = 0; i < pivot.length; i++) {
      x[pivot[i]] = i;
    }
    return x;
  }
}
