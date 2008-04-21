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

import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

/**
 * Doubly sparse matrix. Implemented as a Map of SparseVector rows
 */
/**
 * @author jeff
 * 
 */
/**
 * @author jeff
 * 
 */
public class SparseMatrix extends AbstractMatrix {

  private int[] cardinality;

  private Map<Integer, Vector> rows;

  /**
   * Construct a matrix of the given cardinality with the given row map
   * 
   * @param cardinality
   *            the int[2] cardinality desired
   * @param rows
   *            a Map<Integer, SparseVector> of rows
   */
  public SparseMatrix(int[] cardinality, Map<Integer, SparseVector> rows) {
    this.cardinality = cardinality.clone();
    this.rows = new HashMap<Integer, Vector>();
    for (Integer row : rows.keySet())
      this.rows.put(row, rows.get(row).copy());
  }

  /**
   * Construct a matrix of the given cardinality
   * 
   * @param cardinality
   *            the int[2] cardinality desired
   */
  public SparseMatrix(int[] cardinality) {
    super();
    this.cardinality = cardinality.clone();
    this.rows = new HashMap<Integer, Vector>();
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
    out.append("[s").append(cardinality[ROW]).append(", ");
    for (Integer row : rows.keySet())
      out.append(rows.get(row).asWritableComparable());
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
    SparseMatrix copy = new SparseMatrix(cardinality);
    for (Integer row : rows.keySet())
      copy.rows.put(row, rows.get(row).copy());
    return copy;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#getQuick(int, int)
   */
  @Override
  public double getQuick(int row, int column) {
    Vector r = rows.get(row);
    if (r == null)
      return 0.0;
    else
      return r.getQuick(column);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#haveSharedCells(org.apache.mahout.matrix.Matrix)
   */
  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof SparseMatrix)
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
    return new SparseMatrix(cardinality);
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
    return new SparseMatrix(c);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#setQuick(int, int, double)
   */
  @Override
  public void setQuick(int row, int column, double value) {
    if (rows.get(row) == null)
      rows.put(row, new SparseVector(cardinality[COL]));
    rows.get(row).setQuick(column, value);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.AbstractMatrix#size()
   */
  @Override
  public int[] size() {
    int[] result = new int[2];
    result[ROW] = rows.size();
    for (Integer row : rows.keySet())
      result[COL] = Math.max(result[COL], rows.get(row).size());
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
    if (size[ROW] > cardinality[ROW] || size[COL] > cardinality[COL])
      throw new CardinalityException();
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > cardinality[ROW]
        || offset[COL] < 0 || offset[COL] + size[COL] > cardinality[COL])
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
    for (int row = 0; row < cardinality[ROW]; row++) {
      double val = other.getQuick(row);
      if (val != 0.0) {
        Vector r = rows.get(row);
        if (r == null) {
          r = new SparseVector(cardinality[ROW]);
          rows.put(row, r);
        }
        r.setQuick(column, val);
      }
    }
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
    rows.put(row, other);
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
    Vector res = rows.get(row);
    if (res == null)
      res = new SparseVector(cardinality[ROW]);
    return res;
  }

}
