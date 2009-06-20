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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.NoSuchElementException;

/**
 * Implements vector that only stores non-zero doubles
 */
public class SparseVector extends AbstractVector {

  /** For serialization purposes only. */
  public SparseVector() {
  }

  private OrderedIntDoubleMapping values;

  private int cardinality;

  public static boolean optimizeTimes = true;

  public SparseVector(String name, int cardinality, int size) {
    super(name);
    values = new OrderedIntDoubleMapping(size);
    this.cardinality = cardinality;
  }

  public SparseVector(String name, int cardinality) {
    this(name, cardinality, cardinality / 8); // arbitrary estimate of
                                              // 'sparseness'
  }

  public SparseVector(int cardinality) {
    this(null, cardinality, cardinality / 8); // arbitrary estimate of
                                              // 'sparseness'
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    int[] cardinality = { rows, columns };
    return new SparseRowMatrix(cardinality);
  }

  @Override
  public int size() {
    return cardinality;
  }

  @Override
  public SparseVector clone() {
    SparseVector result = like();
    result.values = (OrderedIntDoubleMapping) values.clone();
    return result;
  }

  @Override
  public double getQuick(int index) {
    return values.get(index);
  }

  @Override
  public void setQuick(int index, double value) {
    values.set(index, value);
  }

  @Override
  public int getNumNondefaultElements() {
    return values.getNumMappings();
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > cardinality)
      throw new CardinalityException();
    if (offset < 0 || offset + length > cardinality)
      throw new IndexException();
    return new VectorView(this, offset, length);
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof SparseVector)
      return other == this;
    else
      return other.haveSharedCells(this);
  }

  @Override
  public SparseVector like() {
    return new SparseVector(cardinality);
  }

  @Override
  public Vector like(int newCardinality) {
    return new SparseVector(newCardinality);
  }

  @Override
  public java.util.Iterator<Vector.Element> iterator() {
    return new Iterator();
  }

  /**
   * Indicate whether the two objects are the same or not. Two
   * {@link org.apache.mahout.matrix.Vector}s can be equal even if the
   * underlying implementation is not equal.
   * 
   * @param o The object to compare
   * @return true if the objects have the same cell values and same name, false
   *         otherwise.
   * 
   *         * @see AbstractVector#strictEquivalence(Vector, Vector)
   * @see AbstractVector#equivalent(Vector, Vector)
   */
  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (!(o instanceof Vector))
      return false;

    Vector that = (Vector) o;
    if (this.size() != that.size())
      return false;

    if (that instanceof SparseVector) {
      return (values == null ? ((SparseVector) that).values == null : values
          .equals(((SparseVector) that).values));
    } else {
      return equivalent(this, that);
    }

  }

  @Override
  public int hashCode() {
    int result = (values != null ? values.hashCode() : 0);
    result = 31 * result + cardinality;
    result = 31 * result + (name == null ? 0 : name.hashCode());
    return result;
  }

  private class Iterator implements java.util.Iterator<Vector.Element> {
    private int offset = 0;

    @Override
    public boolean hasNext() {
      return offset < values.getNumMappings();
    }

    @Override
    public Element next() {
      if (offset < values.getNumMappings()) {
        return new Element(values.getIndices()[offset++]);
      }
      throw new NoSuchElementException();
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public double zSum() {
    double result = 0.0;
    for (double value : values.getValues()) {
      result += value;
    }
    return result;
  }

  @Override
  public double dot(Vector x) {
    if (size() != x.size())
      throw new CardinalityException();
    double result = 0.0;
    for (int index : values.getIndices()) {
      result += values.get(index) * x.getQuick(index);
    }
    return result;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(this.name == null ? "" : this.name);
    dataOutput.writeInt(size());
    dataOutput.writeInt(getNumNondefaultElements());
    for (Vector.Element element : this) {
      if (element.get() != 0.0d) {
        dataOutput.writeInt(element.index());
        dataOutput.writeDouble(element.get());
      }
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    this.name = dataInput.readUTF();
    int cardinality = dataInput.readInt();
    int size = dataInput.readInt();
    OrderedIntDoubleMapping values = new OrderedIntDoubleMapping(size);
    for (int i = 0; i < size; i++) {
      values.set(dataInput.readInt(), dataInput.readDouble());
    }
    this.cardinality = cardinality;
    this.values = values;
  }

  @Override
  public Vector times(double x) {
    Vector result;
    if (optimizeTimes) {
      result = like();
      for (Vector.Element element : this) {
        double value = element.get();
        int index = element.index();
        result.setQuick(index, value * x);
      }
    } else {
      result = clone();
      for (int i = 0; i < result.size(); i++)
        result.setQuick(i, getQuick(i) * x);
    }
    return result;
  }

  @Override
  public Vector times(Vector x) {
    if (size() != x.size())
      throw new CardinalityException();
    Vector result;
    if (optimizeTimes) {
      result = like();
      for (Vector.Element element : this) {
        double value = element.get();
        int index = element.index();
        result.setQuick(index, value * x.getQuick(index));
      }
    } else {
      result = clone();
      for (int i = 0; i < result.size(); i++)
        result.setQuick(i, getQuick(i) * x.getQuick(i));
    }
    return result;
  }

}
