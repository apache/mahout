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

import org.apache.mahout.math.function.IntIntFunction;

/**
 * Matrix View backed by an {@link IntIntFunction}
 */
class FunctionalMatrixView extends AbstractMatrix {

  /**
   * view generator function
   */
  private IntIntFunction gf;
  private boolean denseLike;

  public FunctionalMatrixView(int rows, int columns, IntIntFunction gf) {
    this(rows, columns, gf, false);
  }

  /**
   * @param gf        generator function
   * @param denseLike whether like() should create Dense or Sparse matrix.
   */
  public FunctionalMatrixView(int rows, int columns, IntIntFunction gf, boolean denseLike) {
    super(rows, columns);
    this.gf = gf;
    this.denseLike = denseLike;
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    throw new UnsupportedOperationException("Assignment to a matrix not supported");
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    throw new UnsupportedOperationException("Assignment to a matrix view not supported");
  }

  @Override
  public double getQuick(int row, int column) {
    return gf.apply(row, column);
  }

  @Override
  public Matrix like() {
    return like(rows, columns);
  }

  @Override
  public Matrix like(int rows, int columns) {
    if (denseLike)
      return new DenseMatrix(rows, columns);
    else
      return new SparseMatrix(rows, columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    throw new UnsupportedOperationException("Assignment to a matrix view not supported");
  }


}
