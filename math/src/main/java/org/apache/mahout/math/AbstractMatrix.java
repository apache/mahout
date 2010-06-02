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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.PlusMult;
import org.apache.mahout.math.function.UnaryFunction;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

/** A few universal implementations of convenience functions */
public abstract class AbstractMatrix implements Matrix {

  protected Map<String, Integer> columnLabelBindings;

  protected Map<String, Integer> rowLabelBindings;

  public Iterator<MatrixSlice> iterator() {
    return iterateAll();
  }

  public Iterator<MatrixSlice> iterateAll() {
    return new Iterator<MatrixSlice>() {
      private int slice = 0;

      public boolean hasNext() {
        return slice < numSlices();
      }

      public MatrixSlice next() {
        int i = slice++;
        return new MatrixSlice(slice(i), i);
      }

      public void remove() {
        throw new UnsupportedOperationException("remove() not supported for Matrix iterator");
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
  public int numSlices() {
    return numRows();
  }

  public double get(String rowLabel, String columnLabel) throws IndexException,
      UnboundLabelException {
    if (columnLabelBindings == null || rowLabelBindings == null) {
      throw new UnboundLabelException();
    }
    Integer row = rowLabelBindings.get(rowLabel);
    Integer col = columnLabelBindings.get(columnLabel);
    if (row == null || col == null) {
      throw new UnboundLabelException();
    }

    return get(row, col);
  }

  public Map<String, Integer> getColumnLabelBindings() {
    return columnLabelBindings;
  }

  public Map<String, Integer> getRowLabelBindings() {
    return rowLabelBindings;
  }

  public void set(String rowLabel, double[] rowData) {
    if (columnLabelBindings == null) {
      throw new UnboundLabelException();
    }
    Integer row = rowLabelBindings.get(rowLabel);
    if (row == null) {
      throw new UnboundLabelException();
    }
    set(row, rowData);
  }

  public void set(String rowLabel, int row, double[] rowData) {
    if (rowLabelBindings == null) {
      rowLabelBindings = new HashMap<String, Integer>();
    }
    rowLabelBindings.put(rowLabel, row);
    set(row, rowData);
  }

  public void set(String rowLabel, String columnLabel, double value)
      throws IndexException, UnboundLabelException {
    if (columnLabelBindings == null || rowLabelBindings == null) {
      throw new UnboundLabelException();
    }
    Integer row = rowLabelBindings.get(rowLabel);
    Integer col = columnLabelBindings.get(columnLabel);
    if (row == null || col == null) {
      throw new UnboundLabelException();
    }
    set(row, col, value);
  }

  public void set(String rowLabel, String columnLabel, int row, int column,
                  double value) throws IndexException, UnboundLabelException {
    if (rowLabelBindings == null) {
      rowLabelBindings = new HashMap<String, Integer>();
    }
    rowLabelBindings.put(rowLabel, row);
    if (columnLabelBindings == null) {
      columnLabelBindings = new HashMap<String, Integer>();
    }
    columnLabelBindings.put(columnLabel, column);

    set(row, column, value);
  }

  public void setColumnLabelBindings(Map<String, Integer> bindings) {
    columnLabelBindings = bindings;
  }

  public void setRowLabelBindings(Map<String, Integer> bindings) {
    rowLabelBindings = bindings;
  }

  // index into int[2] for column value
  public static final int COL = 1;

  // index into int[2] for row value
  public static final int ROW = 0;

  public int numRows() {
    return size()[ROW];
  }

  public int numCols() {
    return size()[COL];
  }

  public static Matrix decodeMatrix(String formatString) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    Gson gson = builder.create();
    return gson.fromJson(formatString, Matrix.class);
  }

  public String asFormatString() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, Matrix.class);
  }

  public Matrix assign(double value) {
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, value);
      }
    }
    return this;
  }

  public Matrix assign(double[][] values) {
    int[] c = size();
    if (c[ROW] != values.length) {
      throw new CardinalityException(c[ROW], values.length);
    }
    for (int row = 0; row < c[ROW]; row++) {
      if (c[COL] != values[row].length) {
        throw new CardinalityException(c[COL], values[row].length);
      } else {
        for (int col = 0; col < c[COL]; col++) {
          setQuick(row, col, values[row][col]);
        }
      }
    }
    return this;
  }

  public Matrix assign(Matrix other, BinaryFunction function) {
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

  public Matrix assign(UnaryFunction function) {
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, function.apply(getQuick(row, col)));
      }
    }
    return this;
  }

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
      clone.rowLabelBindings = (Map<String, Integer>) ((HashMap<String, Integer>) rowLabelBindings).clone();
    }
    if (columnLabelBindings != null) {
      clone.columnLabelBindings = (Map<String, Integer>) ((HashMap<String, Integer>) columnLabelBindings).clone();
    }
    return clone;
  }

  public Matrix divide(double x) {
    Matrix result = clone();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, result.getQuick(row, col) / x);
      }
    }
    return result;
  }

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

  public Matrix minus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    Matrix result = clone();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, result.getQuick(row, col)
            - other.getQuick(row, col));
      }
    }
    return result;
  }

  public Matrix plus(double x) {
    Matrix result = clone();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, result.getQuick(row, col) + x);
      }
    }
    return result;
  }

  public Matrix plus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW]) {
      throw new CardinalityException(c[ROW], o[ROW]);
    }
    if (c[COL] != o[COL]) {
      throw new CardinalityException(c[COL], o[COL]);
    }
    Matrix result = clone();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, result.getQuick(row, col)
            + other.getQuick(row, col));
      }
    }
    return result;
  }

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

  public Matrix times(double x) {
    Matrix result = clone();
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        result.setQuick(row, col, result.getQuick(row, col) * x);
      }
    }
    return result;
  }

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

    public boolean isDense() {
      return true;
    }

    public boolean isSequentialAccess() {
      return true;
    }

    @Override
    protected Matrix matrixLike(int rows, int columns) {
      return matrix.like(rows, columns);
    }

    public Iterator<Element> iterator() {
      return new Iterator<Element>() {
        private int i = 0;
        public boolean hasNext() {
          return i < size();
        }

        public Element next() {
          if (i >= size()) {
            throw new NoSuchElementException();
          }
          return getElement(i++);
        }

        public void remove() {
          throw new UnsupportedOperationException("Element removal not supported");
        }
      };
    }

    /**
     * Currently delegates to {@link #iterator()}.
     * TODO: This could be optimized to at least skip empty rows if there are many of them.
     * @return an iterator (currently dense).
     */
    public Iterator<Element> iterateNonZero() {
      return iterator();
    }

    public Element getElement(final int i) {
      return new Element() {
        public double get() {
          return getQuick(i);
        }

        public int index() {
          return i;
        }

        public void set(double value) {
          setQuick(i, value);
        }
      };
    }

    public double getQuick(int index) {
      Vector v = rowToColumn ? matrix.getRow(index) : matrix.getColumn(index);
      return v == null ? 0 : v.getQuick(transposeOffset);
    }

    public void setQuick(int index, double value) {
      Vector v = rowToColumn ? matrix.getRow(index) : matrix.getColumn(index);
      if(v == null) {
        v = newVector(numCols);
        matrix.assignRow(index, v);
      }
      v.setQuick(transposeOffset, value);
    }

    protected Vector newVector(int cardinality) {
      return new DenseVector(cardinality);
    }

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
    public int getNumNondefaultElements() {
      return size();
    }
  }

}
