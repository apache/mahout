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

import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.util.Map;

/** Doubly sparse matrix. Implemented as a Map of RandomAccessSparseVector rows */
public class SparseMatrix extends AbstractMatrix {
  private OpenIntObjectHashMap<Vector> rows;
  
  public SparseMatrix() {
  }
  
  /**
   * Construct a matrix of the given cardinality with the given row map
   * 
   * @param cardinality
   *          the int[2] cardinality desired
   * @param rows
   *          a Map<Integer, RandomAccessSparseVector> of rows
   */
  public SparseMatrix(int[] cardinality,
                      Map<Integer,RandomAccessSparseVector> rows) {
    this.cardinality = cardinality.clone();
    this.rows = new OpenIntObjectHashMap<Vector>();
    for (Map.Entry<Integer,RandomAccessSparseVector> entry : rows.entrySet()) {
      this.rows.put(entry.getKey(), entry.getValue().clone());
    }
  }
  
  /**
   * Construct a matrix of the given cardinality
   * 
   * @param cardinality
   *          the int[2] cardinality desired
   */
  public SparseMatrix(int[] cardinality) {
    this.cardinality = cardinality.clone();
    this.rows = new OpenIntObjectHashMap<Vector>();
  }

  /**
   * Construct a matrix with specified number of rows and columns.
   */
  public SparseMatrix(int rows, int columns) {
    this(new int[]{rows, columns});
  }
  
  @Override
  public Matrix clone() {
    SparseMatrix clone = (SparseMatrix) super.clone();
    clone.cardinality = cardinality.clone();
    clone.rows = rows.clone();
    return clone;
  }
  
  public double getQuick(int row, int column) {
    Vector r = rows.get(row);
    return r == null ? 0.0 : r.getQuick(column);
  }
  
  public Matrix like() {
    return new SparseMatrix(cardinality);
  }
  
  public Matrix like(int rows, int columns) {
    return new SparseMatrix(new int[] {rows, columns});
  }
  
  public void setQuick(int row, int column, double value) {
    Vector r = rows.get(row);
    if (r == null) {
      r = new RandomAccessSparseVector(cardinality[COL]);
      rows.put(row, r);
    }
    r.setQuick(column, value);
  }
  
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rows.size();
    for (Vector vectorEntry : rows.values()) {
      result[COL] = Math.max(result[COL], vectorEntry
          .getNumNondefaultElements());
    }
    return result;
  }
  
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], cardinality[ROW]);
    }
    if (offset[ROW] + size[ROW] > cardinality[ROW]) {
      throw new IndexException(offset[ROW] + size[ROW], cardinality[ROW]);
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], cardinality[COL]);
    }
    if (offset[COL] + size[COL] > cardinality[COL]) {
      throw new IndexException(offset[COL] + size[COL], cardinality[COL]);
    }
    return new MatrixView(this, offset, size);
  }
  
  public Matrix assignColumn(int column, Vector other) {
    if (cardinality[ROW] != other.size()) {
      throw new CardinalityException(cardinality[ROW], other.size());
    }
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    for (int row = 0; row < cardinality[ROW]; row++) {
      double val = other.getQuick(row);
      if (val != 0.0) {
        Vector r = rows.get(row);
        if (r == null) {
          r = new RandomAccessSparseVector(cardinality[COL]);
          rows.put(row, r);
        }
        r.setQuick(column, val);
      }
    }
    return this;
  }
  
  public Matrix assignRow(int row, Vector other) {
    if (cardinality[COL] != other.size()) {
      throw new CardinalityException(cardinality[COL], other.size());
    }
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    rows.put(row, other);
    return this;
  }
  
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException(column, cardinality[COL]);
    }
    double[] d = new double[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++) {
      d[row] = getQuick(row, column);
    }
    return new DenseVector(d);
  }
  
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException(row, cardinality[ROW]);
    }
    Vector res = rows.get(row);
    if (res == null) {
      res = new RandomAccessSparseVector(cardinality[COL]);
    }
    return res;
  }
  
}
