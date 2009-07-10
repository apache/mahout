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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Doubly sparse matrix. Implemented as a Map of SparseVector rows */
public class SparseMatrix extends AbstractMatrix {

  private int[] cardinality;

  private Map<Integer, Vector> rows;

  public SparseMatrix() {
    super();
  }

  /**
   * Construct a matrix of the given cardinality with the given row map
   *
   * @param cardinality the int[2] cardinality desired
   * @param rows        a Map<Integer, SparseVector> of rows
   */
  public SparseMatrix(int[] cardinality, Map<Integer, SparseVector> rows) {
    this.cardinality = cardinality.clone();
    this.rows = new HashMap<Integer, Vector>();
    for (Map.Entry<Integer, SparseVector> entry : rows.entrySet()) {
      this.rows.put(entry.getKey(), entry.getValue().clone());
    }
  }

  /**
   * Construct a matrix of the given cardinality
   *
   * @param cardinality the int[2] cardinality desired
   */
  public SparseMatrix(int[] cardinality) {
    this.cardinality = cardinality.clone();
    this.rows = new HashMap<Integer, Vector>();
  }

  @Override
  public int[] size() {
    return cardinality;
  }

  @Override
  public Matrix clone() {
    SparseMatrix clone = (SparseMatrix) super.clone();
    clone.cardinality = cardinality.clone();
    clone.rows = (Map<Integer, Vector>) ((HashMap<Integer, Vector>) rows).clone();
    return clone;
  }

  @Override
  public double getQuick(int row, int column) {
    Vector r = rows.get(row);
    if (r == null) {
      return 0.0;
    } else {
      return r.getQuick(column);
    }
  }

  @Override
  public boolean haveSharedCells(Matrix other) {
    if (other instanceof SparseMatrix) {
      return other == this;
    } else {
      return other.haveSharedCells(this);
    }
  }

  @Override
  public Matrix like() {
    return new SparseMatrix(cardinality);
  }

  @Override
  public Matrix like(int rows, int columns) {
    return new SparseMatrix(new int[]{rows, columns});
  }

  @Override
  public void setQuick(int row, int column, double value) {
    Integer rowKey = row;
    Vector r = rows.get(rowKey);
    if (r == null) {
      r = new SparseVector(cardinality[COL]);
      rows.put(rowKey, r);
    }
    r.setQuick(column, value);
  }

  @Override
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rows.size();
    for (Map.Entry<Integer, Vector> integerVectorEntry : rows.entrySet()) {
      result[COL] = Math.max(result[COL], integerVectorEntry.getValue()
          .getNumNondefaultElements());
    }
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (size[ROW] > cardinality[ROW] || size[COL] > cardinality[COL]) {
      throw new CardinalityException();
    }
    if (offset[ROW] < 0 || offset[ROW] + size[ROW] > cardinality[ROW]
        || offset[COL] < 0 || offset[COL] + size[COL] > cardinality[COL]) {
      throw new IndexException();
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (other.size() != cardinality[ROW] || column >= cardinality[COL]) {
      throw new CardinalityException();
    }
    for (int row = 0; row < cardinality[ROW]; row++) {
      double val = other.getQuick(row);
      if (val != 0.0) {
        Integer rowKey = row;
        Vector r = rows.get(rowKey);
        if (r == null) {
          r = new SparseVector(cardinality[ROW]);
          rows.put(rowKey, r);
        }
        r.setQuick(column, val);
      }
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (row >= cardinality[ROW] || other.size() != cardinality[COL]) {
      throw new CardinalityException();
    }
    rows.put(row, other);
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    if (column < 0 || column >= cardinality[COL]) {
      throw new IndexException();
    }
    double[] d = new double[cardinality[ROW]];
    for (int row = 0; row < cardinality[ROW]; row++) {
      d[row] = getQuick(row, column);
    }
    return new DenseVector(d);
  }

  @Override
  public Vector getRow(int row) {
    if (row < 0 || row >= cardinality[ROW]) {
      throw new IndexException();
    }
    Vector res = rows.get(row);
    if (res == null) {
      res = new SparseVector(cardinality[ROW]);
    }
    return res;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int[] card = {in.readInt(), in.readInt()};
    this.cardinality = card;
    int rowsize = in.readInt();
    this.rows = new HashMap<Integer, Vector>();
    for (int row = 0; row < rowsize; row++) {
      int key = in.readInt();
      rows.put(key, AbstractVector.readVector(in));
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(cardinality[ROW]);
    out.writeInt(cardinality[COL]);
    out.writeInt(rows.size());
    for (Map.Entry<Integer, Vector> integerVectorEntry : rows.entrySet()) {
      out.writeInt(integerVectorEntry.getKey());
      AbstractVector.writeVector(out, integerVectorEntry.getValue());
    }
  }

}
