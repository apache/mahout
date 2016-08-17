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

import it.unimi.dsi.fastutil.ints.Int2ObjectMap.Entry;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

import java.util.Iterator;
import java.util.Map;

import org.apache.mahout.math.flavor.MatrixFlavor;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.list.IntArrayList;

import com.google.common.collect.AbstractIterator;

/** Doubly sparse matrix. Implemented as a Map of RandomAccessSparseVector rows */
public class SparseMatrix extends AbstractMatrix {

  private Int2ObjectOpenHashMap<Vector> rowVectors;
  
  /**
   * Construct a matrix of the given cardinality with the given row map
   *
   * @param rows no of rows
   * @param columns no of columns
   * @param rowVectors a Map<Integer, RandomAccessSparseVector> of rows
   */
  public SparseMatrix(int rows, int columns, Map<Integer, Vector> rowVectors) {
    this(rows, columns, rowVectors, false);
  }

  public SparseMatrix(int rows, int columns, Map<Integer, Vector> rowVectors, boolean shallow) {

    // Why this is passing in a map? iterating it is pretty inefficient as opposed to simple lists...
    super(rows, columns);
    this.rowVectors = new Int2ObjectOpenHashMap<>();
    if (shallow) {
      for (Map.Entry<Integer, Vector> entry : rowVectors.entrySet()) {
        this.rowVectors.put(entry.getKey().intValue(), entry.getValue());
      }
    } else {
      for (Map.Entry<Integer, Vector> entry : rowVectors.entrySet()) {
        this.rowVectors.put(entry.getKey().intValue(), entry.getValue().clone());
      }
    }
  }
  
  /**
   * Construct a matrix with specified number of rows and columns.
   */
  public SparseMatrix(int rows, int columns) {
    super(rows, columns);
    this.rowVectors = new Int2ObjectOpenHashMap<>();
  }

  @Override
  public Matrix clone() {
    SparseMatrix clone = new SparseMatrix(numRows(), numCols());
    for (MatrixSlice slice : this) {
      clone.rowVectors.put(slice.index(), slice.clone());
    }
    return clone;
  }

  @Override
  public int numSlices() {
    return rowVectors.size();
  }

  public Iterator<MatrixSlice> iterateNonEmpty() {
    final int[] keys = rowVectors.keySet().toIntArray();
    return new AbstractIterator<MatrixSlice>() {
      private int slice;
      @Override
      protected MatrixSlice computeNext() {
        if (slice >= rowVectors.size()) {
          return endOfData();
        }
        int i = keys[slice];
        Vector row = rowVectors.get(i);
        slice++;
        return new MatrixSlice(row, i);
      }
    };
  }
  
  @Override
  public double getQuick(int row, int column) {
    Vector r = rowVectors.get(row);
    return r == null ? 0.0 : r.getQuick(column);
  }
  
  @Override
  public Matrix like() {
    return new SparseMatrix(rowSize(), columnSize());
  }
  
  @Override
  public Matrix like(int rows, int columns) {
    return new SparseMatrix(rows, columns);
  }
  
  @Override
  public void setQuick(int row, int column, double value) {
    Vector r = rowVectors.get(row);
    if (r == null) {
      r = new RandomAccessSparseVector(columnSize());
      rowVectors.put(row, r);
    }
    r.setQuick(column, value);
  }
  
  @Override
  public int[] getNumNondefaultElements() {
    int[] result = new int[2];
    result[ROW] = rowVectors.size();
    for (Vector row : rowVectors.values()) {
      result[COL] = Math.max(result[COL], row.getNumNondefaultElements());
    }
    return result;
  }
  
  @Override
  public Matrix viewPart(int[] offset, int[] size) {
    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], rowSize());
    }
    if (offset[ROW] + size[ROW] > rowSize()) {
      throw new IndexException(offset[ROW] + size[ROW], rowSize());
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], columnSize());
    }
    if (offset[COL] + size[COL] > columnSize()) {
      throw new IndexException(offset[COL] + size[COL], columnSize());
    }
    return new MatrixView(this, offset, size);
  }

  @Override
  public Matrix assign(Matrix other, DoubleDoubleFunction function) {
    //TODO generalize to other kinds of functions
    if (Functions.PLUS.equals(function) && other instanceof SparseMatrix) {
      int rows = rowSize();
      if (rows != other.rowSize()) {
        throw new CardinalityException(rows, other.rowSize());
      }
      int columns = columnSize();
      if (columns != other.columnSize()) {
        throw new CardinalityException(columns, other.columnSize());
      }

      SparseMatrix otherSparse = (SparseMatrix) other;
      for(ObjectIterator<Entry<Vector>> fastIterator = otherSparse.rowVectors.int2ObjectEntrySet().fastIterator();
              fastIterator.hasNext();) {
        final Entry<Vector> entry = fastIterator.next();
        final int rowIndex = entry.getIntKey();
        Vector row = rowVectors.get(rowIndex);
        if (row == null) {
          rowVectors.put(rowIndex, entry.getValue().clone());
        } else {
          row.assign(entry.getValue(), Functions.PLUS);
        }
      }
      return this;
    } else {
      return super.assign(other, function);
    }
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
      double val = other.getQuick(row);
      if (val != 0.0) {
        Vector r = rowVectors.get(row);
        if (r == null) {
          r = new RandomAccessSparseVector(columnSize());
          rowVectors.put(row, r);
        }
        r.setQuick(column, val);
      }
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
    rowVectors.put(row, other);
    return this;
  }
  
  @Override
  public Vector viewRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    Vector res = rowVectors.get(row);
    if (res == null) {
      res = new RandomAccessSparseVector(columnSize());
      rowVectors.put(row, res);
    }
    return res;
  }

  /** special method necessary for efficient serialization */
  public IntArrayList nonZeroRowIndices() {
    return new IntArrayList(rowVectors.keySet().toIntArray());
  }

  @Override
  public MatrixFlavor getFlavor() {
    return MatrixFlavor.SPARSEROWLIKE;
  }
}
