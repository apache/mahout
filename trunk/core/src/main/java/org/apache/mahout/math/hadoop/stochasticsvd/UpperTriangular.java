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

package org.apache.mahout.math.hadoop.stochasticsvd;

import org.apache.mahout.math.AbstractMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * 
 * Quick and dirty implementation of some {@link Matrix} methods 
 * over packed upper triangular matrix.
 *
 */
public class UpperTriangular extends AbstractMatrix {

  private static final double EPSILON = 1.0e-12; // assume anything less than
                                                 // that to be 0 during
                                                 // non-upper assignments

  private double[] values;
  private int n;

  /**
   * represents n x n upper triangular matrix
   * 
   * @param n
   */

  public UpperTriangular(int n) {
    values = new double[n * (n + 1) / 2];
    this.n = n;
    cardinality[0] = cardinality[1] = n;
  }

  public UpperTriangular(Vector data) {
    n = (int) Math.round((-1 + Math.sqrt(1 + 8 * data.size())) / 2);
    cardinality[0] = cardinality[1] = n;
    values = new double[n * (n + 1) / 2];
    int n = data.size();
    // if ( data instanceof DenseVector )
    // ((DenseVector)data).
    // system.arraycopy would've been much faster, but this way it's a drag
    // on B-t job.
    for (int i = 0; i < n; i++) {
      values[i] = data.getQuick(i);
    }
  }

  public UpperTriangular(double[] data, boolean shallow) {
    if (data == null) {
      throw new IllegalArgumentException("data");
    }
    values = shallow ? data : data.clone();
    n = (int) Math.round((-1 + Math.sqrt(1 + 8 * data.length)) / 2);
    cardinality[0] = cardinality[1] = n;
  }

  // copy-constructor
  public UpperTriangular(UpperTriangular mx) {
    this(mx.values, false);
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {

    throw new UnsupportedOperationException();
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    for (int i = 0; i < row; i++) {
      if (other.getQuick(i) > EPSILON) {
        throw new IllegalArgumentException("non-triangular source");
      }
    }
    for (int i = row; i < n; i++) {
      setQuick(row, i, other.get(i));
    }
    return this;
  }

  public Matrix assignRow(int row, double[] other) {
    System.arraycopy(other, row, values, getL(row, row), n - row);
    return this;
  }

  @Override
  public Vector getColumn(int column) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vector getRow(int row) {
    throw new UnsupportedOperationException();
  }

  @Override
  public double getQuick(int row, int column) {
    if (row > column) {
      return 0;
    }
    return values[getL(row, column)];
  }

  private int getL(int row, int col) {
    return (((n << 1) - row + 1) * row >> 1) + col - row;
  }

  @Override
  public Matrix like() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Matrix like(int rows, int columns) {
    throw new UnsupportedOperationException();
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
    throw new UnsupportedOperationException();
  }

  double[] getData() {
    return values;
  }

}
