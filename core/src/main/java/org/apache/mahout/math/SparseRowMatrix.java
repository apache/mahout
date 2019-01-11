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

import org.apache.mahout.math.flavor.MatrixFlavor;
import org.apache.mahout.math.flavor.TraversingStructureEnum;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;

/**
 * sparse matrix with general element values whose rows are accessible quickly. Implemented as a row
 * array of either SequentialAccessSparseVectors or RandomAccessSparseVectors.
 */
public class SparseRowMatrix extends AbstractMatrix {
  private Vector[] rowVectors;

  private final boolean randomAccessRows;

  private static final Logger log = LoggerFactory.getLogger(SparseRowMatrix.class);

  /**
   * Construct a sparse matrix starting with the provided row vectors.
   *
   * @param rows       The number of rows in the result
   * @param columns    The number of columns in the result
   * @param rowVectors a Vector[] array of rows
   */
  public SparseRowMatrix(int rows, int columns, Vector[] rowVectors) {
    this(rows, columns, rowVectors, false, rowVectors instanceof RandomAccessSparseVector[]);
  }

  public SparseRowMatrix(int rows, int columns, boolean randomAccess) {
    this(rows, columns, randomAccess
        ? new RandomAccessSparseVector[rows]
        : new SequentialAccessSparseVector[rows],
      true,
      randomAccess);
  }

  public SparseRowMatrix(int rows, int columns, Vector[] vectors, boolean shallowCopy, boolean randomAccess) {
    super(rows, columns);
    this.randomAccessRows = randomAccess;
    this.rowVectors = vectors.clone();
    for (int row = 0; row < rows; row++) {
      if (vectors[row] == null) {
        // TODO: this can't be right to change the argument
        vectors[row] = randomAccess
          ? new RandomAccessSparseVector(numCols(), 10)
          : new SequentialAccessSparseVector(numCols(), 10);
      }
      this.rowVectors[row] = shallowCopy ? vectors[row] : vectors[row].clone();
    }
  }

  /**
   * Construct a matrix of the given cardinality, with rows defaulting to RandomAccessSparseVector
   * implementation
   *
   * @param rows      Number of rows in result
   * @param columns   Number of columns in result
   */
  public SparseRowMatrix(int rows, int columns) {
    this(rows, columns, true);
  }

  @Override
  public Matrix clone() {
    SparseRowMatrix clone = (SparseRowMatrix) super.clone();
    clone.rowVectors = new Vector[rowVectors.length];
    for (int i = 0; i < rowVectors.length; i++) {
      clone.rowVectors[i] = rowVectors[i].clone();
    }
    return clone;
  }

  @Override
  public double getQuick(int row, int column) {
    return rowVectors[row] == null ? 0.0 : rowVectors[row].getQuick(column);
  }

  @Override
  public Matrix like() {
    return new SparseRowMatrix(rowSize(), columnSize(), randomAccessRows);
  }

  @Override
  public Matrix like(int rows, int columns) {
    return new SparseRowMatrix(rows, columns, randomAccessRows);
  }

  @Override
  public void setQuick(int row, int column, double value) {
    rowVectors[row].setQuick(column, value);
  }

  @Override
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rowVectors.length;
    for (int row = 0; row < rowSize(); row++) {
      result[COL] = Math.max(result[COL], rowVectors[row].getNumNondefaultElements());
    }
    return result;
  }

  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], rowVectors.length);
    }
    if (offset[ROW] + size[ROW] > rowVectors.length) {
      throw new IndexException(offset[ROW] + size[ROW], rowVectors.length);
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], rowVectors[ROW].size());
    }
    if (offset[COL] + size[COL] > rowVectors[ROW].size()) {
      throw new IndexException(offset[COL] + size[COL], rowVectors[ROW].size());
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assign(Matrix other, DoubleDoubleFunction function) {
    int rows = rowSize();
    if (rows != other.rowSize()) {
      throw new CardinalityException(rows, other.rowSize());
    }
    int columns = columnSize();
    if (columns != other.columnSize()) {
      throw new CardinalityException(columns, other.columnSize());
    }
    for (int row = 0; row < rows; row++) {
      try {
        Iterator<Vector.Element> sparseRowIterator = ((SequentialAccessSparseVector) this.rowVectors[row])
                .iterateNonZero();
        if (function.isLikeMult()) { // TODO: is this a sufficient test?
          // TODO: this may cause an exception if the row type is not compatible but it is currently guaranteed to be
          // a SequentialAccessSparseVector, should "try" here just in case and Warn
          // TODO: can we use iterateNonZero on both rows until the index is the same to get better speedup?

          // TODO: SASVs have an iterateNonZero that returns zeros, this should not hurt but is far from optimal
          // this might perform much better if SparseRowMatrix were backed by RandomAccessSparseVectors, which
          // are backed by fastutil hashmaps and the iterateNonZero actually does only return nonZeros.
          while (sparseRowIterator.hasNext()) {
            Vector.Element element = sparseRowIterator.next();
            int col = element.index();
            setQuick(row, col, function.apply(element.get(), other.getQuick(row, col)));
          }
        } else {
          for (int col = 0; col < columns; col++) {
            setQuick(row, col, function.apply(getQuick(row, col), other.getQuick(row, col)));
          }
        }

      } catch (ClassCastException e) {
        // Warn and use default implementation
        log.warn("Error casting the row to SequentialAccessSparseVector, this should never happen because" +
                "SparseRomMatrix is always made of SequentialAccessSparseVectors. Proceeding with non-optimzed" +
                "implementation.");
        for (int col = 0; col < columns; col++) {
          setQuick(row, col, function.apply(getQuick(row, col), other.getQuick(row, col)));
        }
      }
    }
    return this;
  }

  @Override
  public Matrix assignColumn(int column, Vector other) {
    if (rowSize() != other.size()) {
      throw new CardinalityException(rowSize(), other.size());
    }
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    for (int row = 0; row < rowSize(); row++) {
      rowVectors[row].setQuick(column, other.getQuick(row));
    }
    return this;
  }

  @Override
  public Matrix assignRow(int row, Vector other) {
    if (columnSize() != other.size()) {
      throw new CardinalityException(columnSize(), other.size());
    }
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    rowVectors[row].assign(other);
    return this;
  }

  /**
   * @param row an int row index
   * @return a shallow view of the Vector at specified row (ie you may mutate the original matrix
   * using this row)
   */
  @Override
  public Vector viewRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    return rowVectors[row];
  }

  @Override
  public Matrix transpose() {
    SparseColumnMatrix scm = new SparseColumnMatrix(columns, rows);
    for (int i = 0; i < rows; i++) {
      Vector row = rowVectors[i];
      if (row.getNumNonZeroElements() > 0) {
        scm.assignColumn(i, row);
      }
    }
    return scm;
  }

  @Override
  public Matrix times(Matrix other) {
    if (columnSize() != other.rowSize()) {
      throw new CardinalityException(columnSize(), other.rowSize());
    }

    if (other instanceof SparseRowMatrix) {
      SparseRowMatrix y = (SparseRowMatrix) other;
      SparseRowMatrix result = (SparseRowMatrix) like(rowSize(), other.columnSize());

      for (int i = 0; i < rows; i++) {
        Vector row = rowVectors[i];
        for (Vector.Element element : row.nonZeroes()) {
          result.rowVectors[i].assign(y.rowVectors[element.index()], Functions.plusMult(element.get()));
        }
      }
      return result;
    } else {
      if (other.viewRow(0).isDense()) {
        // result is dense, but can be computed relatively cheaply
        Matrix result = other.like(rowSize(), other.columnSize());

        for (int i = 0; i < rows; i++) {
          Vector row = rowVectors[i];
          Vector r = new DenseVector(other.columnSize());
          for (Vector.Element element : row.nonZeroes()) {
            r.assign(other.viewRow(element.index()), Functions.plusMult(element.get()));
          }
          result.viewRow(i).assign(r);
        }
        return result;
      } else {
        // other is sparse, but not something we understand intimately
        SparseRowMatrix result = (SparseRowMatrix) like(rowSize(), other.columnSize());

        for (int i = 0; i < rows; i++) {
          Vector row = rowVectors[i];
          for (Vector.Element element : row.nonZeroes()) {
            result.rowVectors[i].assign(other.viewRow(element.index()), Functions.plusMult(element.get()));
          }
        }
        return result;
      }
    }
  }

  @Override
  public MatrixFlavor getFlavor() {
    return MatrixFlavor.SPARSELIKE;
  }
}
