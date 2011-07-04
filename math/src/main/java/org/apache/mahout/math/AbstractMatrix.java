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

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Maps;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.VectorFunction;

import java.util.Iterator;
import java.util.Map;

/** A few universal implementations of convenience functions */
public abstract class AbstractMatrix implements Matrix {

  protected Map<String, Integer> columnLabelBindings;

  protected Map<String, Integer> rowLabelBindings;

  protected int[] cardinality = new int[2];

  @Override
  public int columnSize() {
    return cardinality[COL];
  }

  @Override
  public int rowSize() {
    return cardinality[ROW];
  }

  @Override
  public int[] size() {
    return cardinality;
  }

  @Override
  public Iterator<MatrixSlice> iterator() {
    return iterateAll();
  }

  @Override
  public Iterator<MatrixSlice> iterateAll() {
    return new AbstractIterator<MatrixSlice>() {
      private int slice;
      @Override
      protected MatrixSlice computeNext() {
        if (slice >= numSlices()) {
          return endOfData();
        }
        int i = slice++;
        return new MatrixSlice(slice(i), i);
      }
    };
  }

  /**
   * Abstracted out for iterating over either rows or columns (default is rows).
   * @param index the row or column number to grab as a vector (shallowly)
   * @return the row or column vector at that index.
   */
  protected Vector slice(int index) {
    return getRow(index);
  }

  /**
   * Abstracted out for the iterator
   * @return numRows() for row-based iterator, numColumns() for column-based.
   */
  @Override
  public int numSlices() {
    return numRows();
  }

  @Override
  public double get(String rowLabel, String columnLabel) {
    if (columnLabelBindings == null || rowLabelBindings == null) {
      throw new IllegalStateException("Unbound label");
    }
    Integer row = rowLabelBindings.get(rowLabel);
    Integer col = columnLabelBindings.get(columnLabel);
    if (row == null || col == null) {
      throw new IllegalStateException("Unbound label");
    }

    return get(row, col);
  }

  @Override
  public Map<String, Integer> getColumnLabelBindings() {
    return columnLabelBindings;
  }

  @Override
  public Map<String, Integer> getRowLabelBindings() {
    return rowLabelBindings;
  }

  @Override
  public void set(String rowLabel, double[] rowData) {
    if (columnLabelBindings == null) {
      throw new IllegalStateException("Unbound label");
    }
    Integer row = rowLabelBindings.get(rowLabel);
    if (row == null) {
      throw new IllegalStateException("Unbound label");
    }
    set(row, rowData);
  }

  @Override
  public void set(String rowLabel, int row, double[] rowData) {
    if (rowLabelBindings == null) {
      rowLabelBindings = Maps.newHashMap();
    }
    rowLabelBindings.put(rowLabel, row);
    set(row, rowData);
  }

  @Override
  public void set(String rowLabel, String columnLabel, double value) {
    if (columnLabelBindings == null || rowLabelBindings == null) {
      throw new IllegalStateException("Unbound label");
    }
    Integer row = rowLabelBindings.get(rowLabel);
    Integer col = columnLabelBindings.get(columnLabel);
    if (row == null || col == null) {
      throw new IllegalStateException("Unbound label");
    }
    set(row, col, value);
  }

  @Override
  public void set(String rowLabel, String columnLabel, int row, int column, double value) {
    if (rowLabelBindings == null) {
      rowLabelBindings = Maps.newHashMap();
    }
    rowLabelBindings.put(rowLabel, row);
    if (columnLabelBindings == null) {
      columnLabelBindings = Maps.newHashMap();
    }
    columnLabelBindings.put(columnLabel, column);

    set(row, column, value);
  }

  @Override
  public void setColumnLabelBindings(Map<String, Integer> bindings) {
    columnLabelBindings = bindings;
  }

  @Override
  public void setRowLabelBindings(Map<String, Integer> bindings) {
    rowLabelBindings = bindings;
  }

  // index into int[2] for column value
  public static final int COL = 1;

  // index into int[2] for row value
  public static final int ROW = 0;

  @Override
  public int numRows() {
    return size()[ROW];
  }

  @Override
  public int numCols() {
    return size()[COL];
  }

  @Override
  public String asFormatString() {
    return toString();
  }

  @Override
  public Matrix assign(double value) {
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, value);
      }
    }
    return this;
  }

  @Override
  public Matrix assign(double[][] values) {
    int[] c = size();
    if (c[ROW] != values.length) {
      throw new CardinalityException(c[ROW], values.length);
    }
    for (int row = 0; row < c[ROW]; row++) {
      if (c[COL] == values[row].length) {
        for (int col = 0; col < c[COL]; col++) {
          setQuick(row, col, values[row][col]);
        }
      } else {
        throw new CardinalityException(c[COL], values[row].length);
      }
    }
    return this;
  }

  @Override
  public Matrix assign(Matrix other, DoubleDoubleFunction function) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, function.apply(getQuick(row, col), other.getQuick(
            row, col)));
      }
    }
    return this;
  }

  @Override
  public Matrix assign(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, other.getQuick(row, col));
      }
    }
    return this;
  }

  @Override
  public Matrix assign(DoubleFunction function) {
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, function.apply(getQuick(row, col)));
      }
    }
    return this;
  }

  /**
   * Collects the results of a function applied to each row of a matrix.
   *
   * @param f The function to be applied to each row.
   * @return The vector of results.
   */
  @Override
  public Vector aggregateRows(VectorFunction f) {
    Vector r = new DenseVector(numRows());
    int n = numRows();
    for (int row = 0; row < n; row++) {
      r.set(row, f.apply(viewRow(row)));
    }
    return r;
  }

  /**
   * Returns a view of a row.  Changes to the view will affect the original.
   * @param row  Which row to return.
   * @return A vector that references the desired row.
   */
  @Override
  public Vector viewRow(int row) {
    return new MatrixVectorView(this, row, 0, 0, 1);
  }


  /**
   * Returns a view of a row.  Changes to the view will affect the original.
   * @param column Which column to return.
   * @return A vector that references the desired column.
   */
  @Override
  public Vector viewColumn(int column) {
    return new MatrixVectorView(this, 0, column, 1, 0);
  }

  /**
   * Collects the results of a function applied to each column of a matrix.
   *
   * @param f The function to be applied to each column.
   * @return The vector of results.
   */
  @Override
  public Vector aggregateColumns(VectorFunction f) {
    Vector r = new DenseVector(numCols());
    for (int col = 0; col < numCols(); col++) {
      r.set(col, f.apply(viewColumn(col)));
    }
    return r;
  }

  /**
   * Collects the results of a function applied to each element of a matrix and then aggregated.
   *
   * @param combiner A function that combines the results of the mapper.
   * @param mapper   A function to apply to each element.
   * @return The result.
   */
  @Override
  public double aggregate(final DoubleDoubleFunction combiner, final DoubleFunction mapper) {
    return aggregateRows(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.aggregate(combiner, mapper);
      }
    }).aggregate(combiner, Functions.IDENTITY);
  }

  @Override
  public double determinant() {
    int[] card = size();
    int rowSize = card[ROW];
    int columnSize = card[COL];
    if (rowSize != columnSize) {
      throw new CardinalityException(rowSize, columnSize);
    }

    if (rowSize == 2) {
      return getQuick(0, 0) * getQuick(1, 1) - getQuick(0, 1) * getQuick(1, 0);
    } else {
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

  @Override
  public Matrix clone() {
    AbstractMatrix clone;
    try {
      clone = (AbstractMatrix) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new IllegalStateException(cnse); // can't happen
    }
    if (rowLabelBindings != null) {
      clone.rowLabelBindings = Maps.newHashMap(rowLabelBindings);
    }
    if (columnLabelBindings != null) {
      clone.columnLabelBindings = Maps.newHashMap(columnLabelBindings);
    }
    return clone;
  }

  @Override
  public Matrix divide(double x) {
    Matrix result = like();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, getQuick(row, col) / x);
      }
    }
    return result;
  }

  @Override
  public double get(int row, int column) {
    int[] c = size();
    if (row < 0 || row >= c[ROW]) {
      throw new IndexException(row, c[ROW]);
    }
    if (column < 0 || column >= c[COL]) {
      throw new IndexException(column, c[COL]);
    }
    return getQuick(row, column);
  }

  @Override
  public Matrix minus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    Matrix result = like();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, getQuick(row, col)
            - other.getQuick(row, col));
      }
    }
    return result;
  }

  @Override
  public Matrix plus(double x) {
    Matrix result = like();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, getQuick(row, col) + x);
      }
    }
    return result;
  }

  @Override
  public Matrix plus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    Matrix result = like();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, getQuick(row, col)
            + other.getQuick(row, col));
      }
    }
    return result;
  }

  @Override
  public void set(int row, int column, double value) {
    int[] c = size();
    if (row < 0 || row >= c[ROW]) {
      throw new IndexException(row, c[ROW]);
    }
    if (column < 0 || column >= c[COL]) {
      throw new IndexException(column, c[COL]);
    }
    setQuick(row, column, value);
  }

  @Override
  public void set(int row, double[] data) {
    int[] c = size();
    if (c[COL] < data.length) {
      throw new CardinalityException(c[COL], data.length);
    }
    if (row < 0 || row >= c[ROW]) {
      throw new IndexException(row, c[ROW]);
    }

    for (int i = 0; i < c[COL]; i++) {
      setQuick(row, i, data[i]);
    }
  }

  @Override
  public Matrix times(double x) {
    Matrix result = like();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, getQuick(row, col) * x);
      }
    }
    return result;
  }

  @Override
  public Matrix times(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[COL] != o[ROW]) {
      throw new CardinalityException(c[COL], o[ROW]);
    }
    Matrix result = like(c[ROW], o[COL]);
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < o[COL]; col++) {
        double sum = 0;
        for (int k = 0; k < c[COL]; k++) {
          sum += getQuick(row, k) * other.getQuick(k, col);
        }
        result.setQuick(row, col, sum);
      }
    }
    return result;
  }

  @Override
  public Vector times(Vector v) {
    int[] c = size();
    if (c[COL] != v.size()) {
      throw new CardinalityException(c[COL], v.size());
    }
    Vector w = new DenseVector(c[ROW]);
    for (int i = 0; i < c[ROW]; i++) {
      w.setQuick(i, v.dot(getRow(i)));
    }
    return w;
  }

  @Override
  public Vector timesSquared(Vector v) {
    int[] c = size();
    if (c[COL] != v.size()) {
      throw new CardinalityException(c[COL], v.size());
    }
    Vector w = new DenseVector(c[COL]);
    for (int i = 0; i < c[ROW]; i++) {
      Vector xi = getRow(i);
      double d = xi.dot(v);
      if (d != 0.0) {
        w.assign(xi, new PlusMult(d));
      }

    }
    return w;
  }

  @Override
  public Matrix transpose() {
    int[] card = size();
    Matrix result = like(card[COL], card[ROW]);
    for (int row = 0; row < card[ROW]; row++) {
      for (int col = 0; col < card[COL]; col++) {
        result.setQuick(col, row, getQuick(row, col));
      }
    }
    return result;
  }

  @Override
  public Matrix viewPart(int rowOffset, int rowsRequested, int columnOffset, int columnsRequested) {
    return viewPart(new int[]{rowOffset, columnOffset}, new int[]{rowsRequested, columnsRequested});
  }

  @Override
  public double zSum() {
    double result = 0;
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result += getQuick(row, col);
      }
    }
    return result;
  }

  protected class TransposeViewVector extends AbstractVector {

    private final Matrix matrix;
    private final int transposeOffset;
    private final int numCols;
    private final boolean rowToColumn;

    protected TransposeViewVector(Matrix m, int offset) {
      this(m, offset, true);
    }

    protected TransposeViewVector(Matrix m, int offset, boolean rowToColumn) {
      super(rowToColumn ? m.numRows() : m.numCols());
      matrix = m;
      this.transposeOffset = offset;
      this.rowToColumn = rowToColumn;
      numCols = rowToColumn ? m.numCols() : m.numRows();
    }

    @Override
    public Vector clone() {
      Vector v = new DenseVector(size());
      addTo(v);
      return v;
    }

    @Override
    public boolean isDense() {
      return true;
    }

    @Override
    public boolean isSequentialAccess() {
      return true;
    }

    @Override
    protected Matrix matrixLike(int rows, int columns) {
      return matrix.like(rows, columns);
    }

    @Override
    public Iterator<Element> iterator() {
      return new AbstractIterator<Element>() {
        private int i;
        @Override
        protected Element computeNext() {
          if (i >= size()) {
            return endOfData();
          }
          return getElement(i++);
        }
      };
    }

    /**
     * Currently delegates to {@link #iterator()}.
     * TODO: This could be optimized to at least skip empty rows if there are many of them.
     * @return an iterator (currently dense).
     */
    @Override
    public Iterator<Element> iterateNonZero() {
      return iterator();
    }

    @Override
    public Element getElement(final int i) {
      return new Element() {
        @Override
        public double get() {
          return getQuick(i);
        }

        @Override
        public int index() {
          return i;
        }

        @Override
        public void set(double value) {
          setQuick(i, value);
        }
      };
    }

    @Override
    public double getQuick(int index) {
      Vector v = rowToColumn ? matrix.getRow(index) : matrix.getColumn(index);
      return v == null ? 0 : v.getQuick(transposeOffset);
    }

    @Override
    public void setQuick(int index, double value) {
      Vector v = rowToColumn ? matrix.getRow(index) : matrix.getColumn(index);
      if (v == null) {
        v = newVector(numCols);
        matrix.assignRow(index, v);
      }
      v.setQuick(transposeOffset, value);
    }

    protected Vector newVector(int cardinality) {
      return new DenseVector(cardinality);
    }

    @Override
    public Vector like() {
      return new DenseVector(size());
    }

    public Vector like(int cardinality) {
      return new DenseVector(cardinality);
    }

    /**
     * TODO: currently I don't know of an efficient way to getVector this value correctly.
     *
     * @return the number of nonzero entries
     */
    @Override
    public int getNumNondefaultElements() {
      return size();
    }
  }

}
