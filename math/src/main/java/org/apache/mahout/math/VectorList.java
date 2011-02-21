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

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Provides a very flexible matrix that is based on a simple list of vectors.
 */
public class VectorList extends AbstractMatrix {
  private final int columns;
  private List<Vector> data = Lists.newArrayList();

  public VectorList(int columns) {
    this.columns = columns;
    cardinality[COL] = columns;
  }

  public VectorList(int rows, int columns) {
    this(columns);
    extendTo(rows);
  }

  @Override
  public int columnSize() {
    return columns;
  }

  @Override
  public int rowSize() {
    return data.size();
  }

  @Override
  public int[] size() {
    cardinality[ROW] = data.size();
    return cardinality;
  }

  public Matrix assignColumn(int column, Vector other) {
    if (other.size() != rowSize()) {
      throw new CardinalityException(rowSize(), other.size());
    }
    int i = 0;
    for (Vector row : data) {
      if (row == null) {
        throw new NullPointerException("Can't insert value into null row ... is matrix row sparse?");
      }
      row.set(column, other.get(i));
      i++;
    }
    return this;
  }

  public Matrix assignRow(int row, Vector other) {
    if (other.size() != columns) {
      throw new CardinalityException(columns, other.size());
    }
    extendTo(row + 1);
    data.set(row, other);
    return this;
  }

  public Vector getColumn(final int column) {
    if (column < 0 || column >= columnSize()) {
      throw new IndexException(column, columnSize());
    }
    return new AbstractVector(rowSize()) {
      @Override
      protected Matrix matrixLike(int rows, int columns) {
        throw new UnsupportedOperationException("Can't get a matrix like a VectorList");
      }

      public boolean isDense() {
        return true;
      }

      public boolean isSequentialAccess() {
        return true;
      }

      public Iterator<Element> iterator() {
        return new AbstractIterator<Element>() {
          int i = 0;

          @Override
          protected Element computeNext() {
            if (i >= data.size()) {
              return endOfData();
            } else {
              return new Element() {
                final int row = i++;

                public double get() {
                  return VectorList.this.get(row, column);
                }

                public int index() {
                  return row;
                }

                public void set(double value) {
                  VectorList.this.setQuick(row, column, value);
                }
              };
            }
          }
        };
      }

      public Iterator<Element> iterateNonZero() {
        return iterator();
      }

      public double getQuick(int index) {
        return VectorList.this.getQuick(index, column);
      }

      public Vector like() {
        return new DenseVector(rowSize());
      }

      public void setQuick(int index, double value) {
        VectorList.this.setQuick(index, column, value);
      }

      public int getNumNondefaultElements() {
        return data.size();
      }
    };
  }

  /**
   * Return the row at the given index
   *
   * @param row an int row index
   * @return a Vector at the index
   * @throws IndexException if the index is out of bounds
   */
  public Vector getRow(int row) {
    if (row < 0 || row >= rowSize()) {
      throw new IndexException(row, rowSize());
    }
    return data.get(row);
  }

  /**
   * Return the value at the given indexes, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  public double getQuick(int row, int column) {
    return data.get(row).getQuick(column);
  }

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  public Matrix like() {
    VectorList r = new VectorList(columns);
    //int i = 0;
    for (Vector vector : data) {
      r.adjoinRow(vector.like());
    }
    return r;
  }

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified
   * size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  public Matrix like(int rows, int columns) {
    VectorList r = new VectorList(rows, columns);
    for (int i = 0; i < rows; i++) {
      r.data.set(i, new DenseVector(columns));
    }
    return r;
  }

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   */
  public void setQuick(int row, int column, double value) {
    data.get(row).setQuick(column, value);
  }

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  public int[] getNumNondefaultElements() {
    return new int[]{data.size(), columns};
  }

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a new Matrix that is a view of the original
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the
   *                              receiver
   */
  public Matrix viewPart(int[] offset, int[] size) {
    cardinality[ROW] = data.size();
    return new MatrixView(this, offset, size);
  }

  private void extendTo(int newLimit) {
    while (data.size() < newLimit) {
      data.add(null);
    }
  }

  public void adjoinRow(Vector vector) {
    Preconditions.checkArgument(vector.size() == columns);
    data.add(vector);
  }

  public void adjoinRow(Matrix other) {
    Preconditions.checkArgument(other.columnSize() == columns);
    for (int row = 0; row < other.rowSize(); row++) {
      adjoinRow(other.getRow(row));
    }
  }

  protected static class JsonVectorListAdapter implements JsonDeserializer<VectorList> {
    private final Type collectionType = new TypeToken<List<Vector>>(){}.getType();
    private final Type labelType = new TypeToken<Map<String, Integer>>(){}.getType();

    public VectorList deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context) {
      JsonObject jo = json.getAsJsonObject();
      VectorList r = new VectorList(jo.get("columns").getAsInt());
      r.data = context.deserialize(jo.get("data"), collectionType);
      if (jo.get("columnLabelBindings") != null) {
        r.columnLabelBindings = context.deserialize(jo.get("columnLabelBindings"), labelType);
      }
      if (jo.get("rowLabelBindings") != null) {
        r.rowLabelBindings = context.deserialize(jo.get("rowLabelBindings"), labelType);
      }

      return r;
    }
  }
}
