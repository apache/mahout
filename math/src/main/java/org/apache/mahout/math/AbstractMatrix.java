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
import org.apache.mahout.math.function.*;

import java.util.Iterator;
import java.util.Map;

/**
 * A few universal implementations of convenience functions
 */
public abstract class AbstractMatrix implements Matrix {

  protected Map<String, Integer> columnLabelBindings;
  protected Map<String, Integer> rowLabelBindings;
  protected int rows;
  protected int columns;

  protected AbstractMatrix(int rows, int columns) {
    this.rows = rows;
    this.columns = columns;
  }

  @Override
  public int columnSize() {
    return columns;
  }

  @Override
  public int rowSize() {
    return rows;
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
        return new MatrixSlice(viewRow(i), i);
      }
    };
  }

  /**
   * Abstracted out for the iterator
   *
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
    return rowSize();
  }

  @Override
  public int numCols() {
    return columnSize();
  }

  @Override
  public String asFormatString() {
    return toString();
  }

  @Override
  public Matrix assign(double value) {
    int rows = rowSize();
    int columns = columnSize();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        setQuick(row, col, value);
      }
    }
    return this;
  }

  @Override
  public Matrix assign(double[][] values) {
    int rows = rowSize();
    if (rows != values.length) {
      throw new CardinalityException(rows, values.length);
    }
    int columns = columnSize();
    for (int row = 0; row < rows; row++) {
      if (columns == values[row].length) {
        for (int col = 0; col < columns; col++) {
          setQuick(row, col, values[row][col]);
        }
      } else {
        throw new CardinalityException(columns, values[row].length);
      }
    }
    return this;
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
      for (int col = 0; col < columns; col++) {
        setQuick(row, col, function.apply(getQuick(row, col), other.getQuick(
          row, col)));
      }
    }
    return this;
  }

  @Override
  public Matrix assign(Matrix other) {
    int rows = rowSize();
    if (rows != other.rowSize()) {
      throw new CardinalityException(rows, other.rowSize());
    }
    int columns = columnSize();
    if (columns != other.columnSize()) {
      throw new CardinalityException(columns, other.columnSize());
    }
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        setQuick(row, col, other.getQuick(row, col));
      }
    }
    return this;
  }

  @Override
  public Matrix assign(DoubleFunction function) {
    int rows = rowSize();
    int columns = columnSize();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
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
   *
   * @param row Which row to return.
   * @return A vector that references the desired row.
   */
  @Override
  public Vector viewRow(int row) {
    return new MatrixVectorView(this, row, 0, 0, 1);
  }


  /**
   * Returns a view of a row.  Changes to the view will affect the original.
   *
   * @param column Which column to return.
   * @return A vector that references the desired column.
   */
  @Override
  public Vector viewColumn(int column) {
    return new MatrixVectorView(this, 0, column, 1, 0);
  }

  /**
   * Provides a view of the diagonal of a matrix.
   */
  @Override
  public Vector viewDiagonal() {
    return new MatrixVectorView(this, 0, 0, 1, 1);
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

  @Override
  public double determinant() {
    int rows = rowSize();
    int columns = columnSize();
    if (rows != columns) {
      throw new CardinalityException(rows, columns);
    }

    if (rows == 2) {
      return getQuick(0, 0) * getQuick(1, 1) - getQuick(0, 1) * getQuick(1, 0);
    } else {
      // TODO: this really should just be one line:
      // TODO: new CholeskyDecomposition(this).getL().viewDiagonal().aggregate(Functions.TIMES)
      int sign = 1;
      double ret = 0;

      for (int i = 0; i < columns; i++) {
        Matrix minor = new DenseMatrix(rows - 1, columns - 1);
        for (int j = 1; j < rows; j++) {
          boolean flag = false; /* column offset flag */
          for (int k = 0; k < columns; k++) {
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

  @SuppressWarnings("CloneDoesntDeclareCloneNotSupportedException")
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
    for (int row = 0; row < rowSize(); row++) {
      for (int col = 0; col < columnSize(); col++) {
        result.setQuick(row, col, getQuick(row, col) / x);
      }
    }
    return result;
  }

  @Override
  public double get(int row, int column) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return getQuick(row, column);
  }

  @Override
  public Matrix minus(Matrix other) {
    int rows = rowSize();
    if (rows != other.rowSize()) {
      throw new CardinalityException(rows, other.rowSize());
    }
    int columns = columnSize();
    if (columns != other.columnSize()) {
      throw new CardinalityException(columns, other.columnSize());
    }
    Matrix result = like();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        result.setQuick(row, col, getQuick(row, col)
          - other.getQuick(row, col));
      }
    }
    return result;
  }

  @Override
  public Matrix plus(double x) {
    Matrix result = like();
    int rows = rowSize();
    int columns = columnSize();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        result.setQuick(row, col, getQuick(row, col) + x);
      }
    }
    return result;
  }

  @Override
  public Matrix plus(Matrix other) {
    int rows = rowSize();
    if (rows != other.rowSize()) {
      throw new CardinalityException(rows, other.rowSize());
    }
    int columns = columnSize();
    if (columns != other.columnSize()) {
      throw new CardinalityException(columns, other.columnSize());
    }
    Matrix result = like();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        result.setQuick(row, col, getQuick(row, col)
          + other.getQuick(row, col));
      }
    }
    return result;
  }

  @Override
  public void set(int row, int column, double value) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    setQuick(row, column, value);
  }

  @Override
  public void set(int row, double[] data) {
    int columns = columnSize();
    if (columns < data.length) {
      throw new CardinalityException(columns, data.length);
    }
    int rows = rowSize();
    if (row < 0 || row >= rows) {
      throw new IndexException(row, rowSize());
    }
    for (int i = 0; i < columns; i++) {
      setQuick(row, i, data[i]);
    }
  }

  @Override
  public Matrix times(double x) {
    Matrix result = like();
    int rows = rowSize();
    int columns = columnSize();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        result.setQuick(row, col, getQuick(row, col) * x);
      }
    }
    return result;
  }

  @Override
  public Matrix times(Matrix other) {
    int columns = columnSize();
    if (columns != other.rowSize()) {
      throw new CardinalityException(columns, other.rowSize());
    }
    int rows = rowSize();
    int otherColumns = other.columnSize();
    Matrix result = like(rows, otherColumns);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < otherColumns; col++) {
        double sum = 0.0;
        for (int k = 0; k < columns; k++) {
          sum += getQuick(row, k) * other.getQuick(k, col);
        }
        result.setQuick(row, col, sum);
      }
    }
    return result;
  }

  @Override
  public Vector times(Vector v) {
    int columns = columnSize();
    if (columns != v.size()) {
      throw new CardinalityException(columns, v.size());
    }
    int rows = rowSize();
    Vector w = new DenseVector(rows);
    for (int row = 0; row < rows; row++) {
      w.setQuick(row, v.dot(viewRow(row)));
    }
    return w;
  }

  @Override
  public Vector timesSquared(Vector v) {
    int columns = columnSize();
    if (columns != v.size()) {
      throw new CardinalityException(columns, v.size());
    }
    int rows = rowSize();
    Vector w = new DenseVector(columns);
    for (int i = 0; i < rows; i++) {
      Vector xi = viewRow(i);
      double d = xi.dot(v);
      if (d != 0.0) {
        w.assign(xi, new PlusMult(d));
      }

    }
    return w;
  }

  @Override
  public Matrix transpose() {
    int rows = rowSize();
    int columns = columnSize();
    Matrix result = like(columns, rows);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
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
  public Matrix viewPart(int[] offset, int[] size) {

    if (offset[ROW] < 0) {
      throw new IndexException(offset[ROW], 0);
    }
    if (offset[ROW] + size[ROW] > rowSize()) {
      throw new IndexException(offset[ROW] + size[ROW], rowSize());
    }
    if (offset[COL] < 0) {
      throw new IndexException(offset[COL], 0);
    }
    if (offset[COL] + size[COL] > columnSize()) {
      throw new IndexException(offset[COL] + size[COL], columnSize());
    }

    return new MatrixView(this, offset, size);
  }


  @Override
  public double zSum() {
    double result = 0;
    for (int row = 0; row < rowSize(); row++) {
      for (int col = 0; col < columnSize(); col++) {
        result += getQuick(row, col);
      }
    }
    return result;
  }

  @Override
  public int[] getNumNondefaultElements() {
    return new int[]{rowSize(), columnSize()};
  }

  protected static class TransposeViewVector extends AbstractVector {

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

    @SuppressWarnings("CloneDoesntCallSuperClone")
    @Override
    public Vector clone() {
      Vector v = new DenseVector(size());
      v.assign(this, Functions.PLUS);
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
     *
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

    /**
     * Used internally by assign() to update multiple indices and values at once.
     * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
     * <p/>
     * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
     *
     * @param updates a mapping of indices to values to merge in the vector.
     */
    @Override
    public void mergeUpdates(OrderedIntDoubleMapping updates) {
      throw new UnsupportedOperationException("Cannot mutate TransposeViewVector");
    }

    @Override
    public double getQuick(int index) {
      Vector v = rowToColumn ? matrix.viewColumn(index) : matrix.viewRow(index);
      return v == null ? 0.0 : v.getQuick(transposeOffset);
    }

    @Override
    public void setQuick(int index, double value) {
      Vector v = rowToColumn ? matrix.viewColumn(index) : matrix.viewRow(index);
      if (v == null) {
        v = newVector(numCols);
        if (rowToColumn) {
          matrix.assignColumn(index, v);
        } else {
          matrix.assignRow(index, v);
        }
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

    @Override
    public double getLookupCost() {
      return (rowToColumn ? matrix.viewColumn(0) : matrix.viewRow(0)).getLookupCost();
    }

    @Override
    public double getIteratorAdvanceCost() {
      return (rowToColumn ? matrix.viewColumn(0) : matrix.viewRow(0)).getIteratorAdvanceCost();
    }

    @Override
    public boolean isAddConstantTime() {
      return (rowToColumn ? matrix.viewColumn(0) : matrix.viewRow(0)).isAddConstantTime();
    }
  }

  @Override
  public String toString() {
    StringBuilder s = new StringBuilder("{\n");
    Iterator<MatrixSlice> it = iterator();
    while (it.hasNext()) {
      MatrixSlice next = it.next();
      s.append("  ").append(next.index()).append("  =>\t").append(next.vector()).append('\n');
    }
    s.append("}");
    return s.toString();
  }
}
