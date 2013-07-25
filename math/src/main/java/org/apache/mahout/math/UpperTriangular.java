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

package org.apache.mahout.math;

/**
 * 
 * Quick and dirty implementation of some {@link org.apache.mahout.math.Matrix} methods
 * over packed upper triangular matrix.
 *
 */
public class UpperTriangular extends AbstractMatrix {

  private static final double EPSILON = 1.0e-12; // assume anything less than
                                                 // that to be 0 during
                                                 // non-upper assignments

  private double[] values;

  /**
   * represents n x n upper triangular matrix
   * 
   * @param n
   */

  public UpperTriangular(int n) {
    super(n, n);
    values = new double[n * (n + 1) / 2];
  }

  public UpperTriangular(double[] data, boolean shallow) {
    this(elementsToMatrixSize(data != null ? data.length : 0));
    if (data == null) {
      throw new IllegalArgumentException("data");
    }
    values = shallow ? data : data.clone();
  }

  public UpperTriangular(Vector data) {
    this(elementsToMatrixSize(data.size()));

    for (Vector.Element el:data.nonZeroes()) {
      values[el.index()] = el.get();
    }
  }

  private static int elementsToMatrixSize(int dataSize) {
    return (int) Math.round((-1 + Math.sqrt(1 + 8 * dataSize)) / 2);
  }

  // copy-constructor
  public UpperTriangular(UpperTriangular mx) {
    this(mx.values, false);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (columnSize() != other.size()) {
      throw new IndexException(columnSize(), other.size());
    }
    if (other.viewPart(column + 1, other.size() - column - 1).norm(1) > 1.0e-14) {
      throw new IllegalArgumentException("Cannot set lower portion of triangular matrix to non-zero");
    }
    for (Vector.Element element : other.viewPart(0, column).all()) {
      setQuick(element.index(), column, element.get());
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (columnSize() != other.size()) {
      throw new IndexException(numCols(), other.size());
    }
    for (int i = 0; i < row; i++) {
      if (Math.abs(other.getQuick(i)) > EPSILON) {
        throw new IllegalArgumentException("non-triangular source");
      }
    }
    for (int i = row; i < rows; i++) {
      setQuick(row, i, other.get(i));
    }
    return this;
  }

  public Matrix assignNonZeroElementsInRow(int row, double[] other) {
    System.arraycopy(other, row, values, getL(row, row), rows - row);
    return this;
  }

  @Override
  public double getQuick(int row, int column) {
    if (row > column) {
      return 0;
    }
    int i = getL(row, column);
    return values[i];
  }

  private int getL(int row, int col) {
    /*
     * each row starts with some zero elements that we don't store. this
     * accumulates an offset of (row+1)*row/2
     */
    return col + row * numCols() - (row + 1) * row / 2;
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
    values[getL(row, column)] = value;
  }

  @Override
  public int[] getNumNondefaultElements() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    return new MatrixView(this, offset, size);
  }

  public double[] getData() {
    return values;
  }

}
