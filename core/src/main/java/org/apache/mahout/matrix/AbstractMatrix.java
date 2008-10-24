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

import org.apache.hadoop.io.WritableComparable;

/**
 * A few universal implementations of convenience functions
 * 
 */
public abstract class AbstractMatrix implements Matrix {

  // index into int[2] for column value
  public static final int COL = 1;

  // index into int[2] for row value
  public static final int ROW = 0;

  public abstract WritableComparable asWritableComparable();

  public abstract String asFormatString();

  public abstract Matrix assignColumn(int column, Vector other);

  public abstract Matrix assignRow(int row, Vector other);

  public abstract int[] cardinality();

  public abstract Matrix copy();

  public abstract Vector getColumn(int column);

  public abstract double getQuick(int row, int column);

  public abstract Vector getRow(int row);

  public abstract boolean haveSharedCells(Matrix other);

  public abstract Matrix like();

  public abstract Matrix like(int rows, int columns);

  public abstract void setQuick(int row, int column, double value);

  public abstract int[] size();

  public abstract double[][] toArray();

  public abstract Matrix viewPart(int[] offset, int[] length);

  public Matrix assign(double value) {
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        setQuick(row, col, value);
    return this;
  }

  public Matrix assign(double[][] values) {
    int[] c = cardinality();
    if (c[ROW] != values.length)
      throw new CardinalityException();
    for (int row = 0; row < c[ROW]; row++)
      if (c[COL] != values[row].length)
        throw new CardinalityException();
      else
        for (int col = 0; col < c[COL]; col++)
          setQuick(row, col, values[row][col]);
    return this;
  }

  public Matrix assign(Matrix other, BinaryFunction function) {
    int[] c = cardinality();
    int[] o = other.cardinality();
    if (c[ROW] != o[ROW] || c[COL] != o[COL])
      throw new CardinalityException();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        setQuick(row, col, function.apply(getQuick(row, col), other.getQuick(
            row, col)));
    return this;
  }

  public Matrix assign(Matrix other) {
    int[] c = cardinality();
    int[] o = other.cardinality();
    if (c[ROW] != o[ROW] || c[COL] != o[COL])
      throw new CardinalityException();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        setQuick(row, col, other.getQuick(row, col));
    return this;
  }

  public Matrix assign(UnaryFunction function) {
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        setQuick(row, col, function.apply(getQuick(row, col)));
    return this;
  }

  public double determinant() {
    int[] card = cardinality();
    int rowSize = card[ROW];
    int columnSize = card[COL];
    if (rowSize != columnSize)
      throw new CardinalityException();

    if (rowSize == 2)
      return getQuick(0, 0) * getQuick(1, 1) - getQuick(0, 1) * getQuick(1, 0);
    else {
      int sign = 1;
      double ret = 0;

      for (int i = 0; i < columnSize; i++) {
        Matrix minor = new DenseMatrix(rowSize - 1, columnSize - 1);
        for (int j = 1; j < rowSize; j++) {
          boolean flag = false; /* column offset flag */
          for (int k = 0; k < columnSize; k++) {
            if (k == i) {
              flag = true;
              continue;
            }
            minor.set(j - 1, flag ? k - 1 : k, getQuick(j, k));
          }
        }
        ret += getQuick(0, i) * sign * minor.determinant();
        sign *= -1;

      }

      return ret;
    }

  }

  public Matrix divide(double x) {
    Matrix result = copy();
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result.setQuick(row, col, result.getQuick(row, col) / x);
    return result;
  }

  public double get(int row, int column) {
    int[] c = cardinality();
    if (row < 0 || column < 0 || row >= c[ROW] || column >= c[COL])
      throw new IndexException();
    return getQuick(row, column);
  }

  public Matrix minus(Matrix other) {
    int[] c = cardinality();
    int[] o = other.cardinality();
    if (c[ROW] != o[ROW] || c[COL] != o[COL])
      throw new CardinalityException();
    Matrix result = copy();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result.setQuick(row, col, result.getQuick(row, col)
            - other.getQuick(row, col));
    return result;
  }

  public Matrix plus(double x) {
    Matrix result = copy();
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result.setQuick(row, col, result.getQuick(row, col) + x);
    return result;
  }

  public Matrix plus(Matrix other) {
    int[] c = cardinality();
    int[] o = other.cardinality();
    if (c[ROW] != o[ROW] || c[COL] != o[COL])
      throw new CardinalityException();
    Matrix result = copy();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result.setQuick(row, col, result.getQuick(row, col)
            + other.getQuick(row, col));
    return result;
  }

  public void set(int row, int column, double value) {
    int[] c = cardinality();
    if (row < 0 || column < 0 || row >= c[ROW] || column >= c[COL])
      throw new IndexException();
    setQuick(row, column, value);
  }

  public void set(int row, double[] data) {
    int[] c = cardinality();
    if (c[COL] < data.length)
      throw new CardinalityException();
    if ((c[ROW] < row) || (row < 0))
      throw new IndexException();

    for (int i = 0; i < c[COL]; i++)
      setQuick(row, i, data[i]);
  }

  public Matrix times(double x) {
    Matrix result = copy();
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result.setQuick(row, col, result.getQuick(row, col) * x);
    return result;
  }

  public Matrix times(Matrix other) {
    int[] c = cardinality();
    int[] o = other.cardinality();
    if (c[COL] != o[ROW])
      throw new CardinalityException();
    Matrix result = like(c[ROW], o[COL]);
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < o[COL]; col++) {
        double sum = 0;
        for (int k = 0; k < c[COL]; k++)
          sum += getQuick(row, k) * other.getQuick(k, col);
        result.setQuick(row, col, sum);
      }
    return result;
  }

  public Matrix transpose() {
    int[] card = cardinality();
    Matrix result = like(card[COL], card[ROW]);
    for (int row = 0; row < card[ROW]; row++)
      for (int col = 0; col < card[COL]; col++)
        result.setQuick(col, row, getQuick(row, col));
    return result;
  }

  public double zSum() {
    double result = 0;
    int[] c = cardinality();
    for (int row = 0; row < c[ROW]; row++)
      for (int col = 0; col < c[COL]; col++)
        result += getQuick(row, col);
    return result;
  }

}
