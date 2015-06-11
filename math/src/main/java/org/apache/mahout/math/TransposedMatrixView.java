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

import org.apache.mahout.math.flavor.BackEnum;
import org.apache.mahout.math.flavor.MatrixFlavor;
import org.apache.mahout.math.flavor.TraversingStructureEnum;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;

/**
 * Matrix View backed by an {@link org.apache.mahout.math.function.IntIntFunction}
 */
class TransposedMatrixView extends AbstractMatrix {

  private Matrix m;

  public TransposedMatrixView(Matrix m) {
    super(m.numCols(), m.numRows());
    this.m = m;
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    m.assignRow(column,other);
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    m.assignColumn(row,other);
    return this;
  }

  @Override
  public double getQuick(int row, int column) {
    return m.getQuick(column,row);
  }

  @Override
  public Matrix like() {
    return m.like(rows, columns);
  }

  @Override
  public Matrix like(int rows, int columns) {
    return m.like(rows,columns);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    m.setQuick(column, row, value);
  }

  @Override
  public Vector viewRow(int row) {
    return m.viewColumn(row);
  }

  @Override
  public Vector viewColumn(int column) {
    return m.viewRow(column);
  }

  @Override
  public Matrix assign(double value) {
    return m.assign(value);
  }

  @Override
  public Matrix assign(Matrix other, DoubleDoubleFunction function) {
    if (other instanceof TransposedMatrixView) {
      m.assign(((TransposedMatrixView) other).m, function);
    } else {
      m.assign(new TransposedMatrixView(other), function);
    }
    return this;
  }

  @Override
  public Matrix assign(Matrix other) {
    if (other instanceof TransposedMatrixView) {
      return m.assign(((TransposedMatrixView) other).m);
    } else {
      return m.assign(new TransposedMatrixView(other));
    }
  }

  @Override
  public Matrix assign(DoubleFunction function) {
    return m.assign(function);
  }

  @Override
  public MatrixFlavor getFlavor() {
    return flavor;
  }

  private MatrixFlavor flavor = new MatrixFlavor() {
    @Override
    public BackEnum getBacking() {
      return m.getFlavor().getBacking();
    }

    @Override
    public TraversingStructureEnum getStructure() {
      TraversingStructureEnum flavor = m.getFlavor().getStructure();
      switch (flavor) {
        case COLWISE:
          return TraversingStructureEnum.ROWWISE;
        case SPARSECOLWISE:
          return TraversingStructureEnum.SPARSEROWWISE;
        case ROWWISE:
          return TraversingStructureEnum.COLWISE;
        case SPARSEROWWISE:
          return TraversingStructureEnum.SPARSECOLWISE;
        default:
          return flavor;
      }
    }

    @Override
    public boolean isDense() {
      return m.getFlavor().isDense();
    }
  };

  Matrix getDelegate() {
    return m;
  }

}
