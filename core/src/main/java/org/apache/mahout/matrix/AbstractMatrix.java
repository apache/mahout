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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

/** A few universal implementations of convenience functions */
public abstract class AbstractMatrix implements Matrix {

  private Map<String, Integer> columnLabelBindings;

  private Map<String, Integer> rowLabelBindings;

  @Override
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
      throw new UnboundLabelException();
    }
    Integer row = rowLabelBindings.get(rowLabel);
    if (row == null) {
      throw new UnboundLabelException();
    }
    set(row, rowData);
  }

  @Override
  public void set(String rowLabel, int row, double[] rowData) {
    if (rowLabelBindings == null) {
      rowLabelBindings = new HashMap<String, Integer>();
    }
    rowLabelBindings.put(rowLabel, row);
    set(row, rowData);
  }

  @Override
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

  @Override
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

  public static Matrix decodeMatrix(String formatString) {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    Type matrixType = new TypeToken<Matrix>() {
    }.getType();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    builder.registerTypeAdapter(matrixType, new JsonMatrixAdapter());
    Gson gson = builder.create();
    return gson.fromJson(formatString, matrixType);
  }

  @Override
  public String asFormatString() {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    Type matrixType = new TypeToken<Matrix>() {
    }.getType();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    builder.registerTypeAdapter(matrixType, new JsonMatrixAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, matrixType);
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
      throw new CardinalityException();
    }
    for (int row = 0; row < c[ROW]; row++) {
      if (c[COL] != values[row].length) {
        throw new CardinalityException();
      } else {
        for (int col = 0; col < c[COL]; col++) {
          setQuick(row, col, values[row][col]);
        }
      }
    }
    return this;
  }

  @Override
  public Matrix assign(Matrix other, BinaryFunction function) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW] || c[COL] != o[COL]) {
      throw new CardinalityException();
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
    if (c[ROW] != o[ROW] || c[COL] != o[COL]) {
      throw new CardinalityException();
    }
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, other.getQuick(row, col));
      }
    }
    return this;
  }

  @Override
  public Matrix assign(UnaryFunction function) {
    int[] c = size();
    for (int row = 0; row < c[ROW]; row++) {
      for (int col = 0; col < c[COL]; col++) {
        setQuick(row, col, function.apply(getQuick(row, col)));
      }
    }
    return this;
  }

  @Override
  public double determinant() {
    int[] card = size();
    int rowSize = card[ROW];
    int columnSize = card[COL];
    if (rowSize != columnSize) {
      throw new CardinalityException();
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

  @Override
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

  @Override
  public double get(int row, int column) {
    int[] c = size();
    if (row < 0 || column < 0 || row >= c[ROW] || column >= c[COL]) {
      throw new IndexException();
    }
    return getQuick(row, column);
  }

  @Override
  public Matrix minus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW] || c[COL] != o[COL]) {
      throw new CardinalityException();
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

  @Override
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

  @Override
  public Matrix plus(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[ROW] != o[ROW] || c[COL] != o[COL]) {
      throw new CardinalityException();
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

  @Override
  public void set(int row, int column, double value) {
    int[] c = size();
    if (row < 0 || column < 0 || row >= c[ROW] || column >= c[COL]) {
      throw new IndexException();
    }
    setQuick(row, column, value);
  }

  @Override
  public void set(int row, double[] data) {
    int[] c = size();
    if (c[COL] < data.length) {
      throw new CardinalityException();
    }
    if ((c[ROW] < row) || (row < 0)) {
      throw new IndexException();
    }

    for (int i = 0; i < c[COL]; i++) {
      setQuick(row, i, data[i]);
    }
  }

  @Override
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

  @Override
  public Matrix times(Matrix other) {
    int[] c = size();
    int[] o = other.size();
    if (c[COL] != o[ROW]) {
      throw new CardinalityException();
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

  @Override
  public void readFields(DataInput in) throws IOException {
    // read the label bindings
    int colSize = in.readInt();
    if (colSize > 0) {
      columnLabelBindings = new HashMap<String, Integer>();
      for (int i = 0; i < colSize; i++) {
        columnLabelBindings.put(in.readUTF(), in.readInt());
      }
    }
    int rowSize = in.readInt();
    if (rowSize > 0) {
      rowLabelBindings = new HashMap<String, Integer>();
      for (int i = 0; i < rowSize; i++) {
        rowLabelBindings.put(in.readUTF(), in.readInt());
      }
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    // write the label bindings
    if (columnLabelBindings == null) {
      out.writeInt(0);
    } else {
      out.writeInt(columnLabelBindings.size());
      for (Map.Entry<String, Integer> stringIntegerEntry : columnLabelBindings.entrySet()) {
        out.writeUTF(stringIntegerEntry.getKey());
        out.writeInt(stringIntegerEntry.getValue());
      }
    }
    if (rowLabelBindings == null) {
      out.writeInt(0);
    } else {
      out.writeInt(rowLabelBindings.size());
      for (Map.Entry<String, Integer> stringIntegerEntry : rowLabelBindings.entrySet()) {
        out.writeUTF(stringIntegerEntry.getKey());
        out.writeInt(stringIntegerEntry.getValue());
      }
    }
  }

  /** Reads a typed Matrix instance from the input stream */
  public static Matrix readMatrix(DataInput in) throws IOException {
    String matrixClassName = in.readUTF();
    Matrix matrix;
    try {
      matrix = Class.forName(matrixClassName).asSubclass(Matrix.class)
          .newInstance();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
    matrix.readFields(in);
    return matrix;
  }

  /** Writes a typed Matrix instance to the output stream */
  public static void writeMatrix(DataOutput out, Matrix matrix)
      throws IOException {
    out.writeUTF(matrix.getClass().getName());
    matrix.write(out);
  }

}
