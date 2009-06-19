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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Arrays;

/**
 * Implements vector as an array of doubles
 */
public class DenseVector extends AbstractVector {

  /** For serialization purposes only */
  public DenseVector() {
  }

  public DenseVector(String name){
    super(name);
  }

  private double[] values;

  /**
   * Construct a new instance using provided values
   * 
   * @param values
   */
  public DenseVector(double[] values) {
    this.values = values.clone();
  }

  public DenseVector(String name, double[] values) {
    super(name);
    this.values = values.clone();
  }

  /**
   * Construct a new instance of the given cardinality
   * 
   * @param cardinality
   */
  public DenseVector(int cardinality) {
    this(null, cardinality);
  }

  public DenseVector(String name, int cardinality) {
    super(name);
    this.values = new double[cardinality];
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  @Override
  public WritableComparable<?> asWritableComparable() {
    return new Text(asFormatString());
  }

  @Override
  public int size() {
    return values.length;
  }

  @Override
  public DenseVector clone() {
    DenseVector denseVector = new DenseVector(values);
    denseVector.setLabelBindings(getLabelBindings());
    return denseVector;
  }

  @Override
  public double getQuick(int index) {
    return values[index];
  }

  @Override
  public DenseVector like() {
    DenseVector denseVector = new DenseVector(size());
    denseVector.setLabelBindings(getLabelBindings());
    return denseVector;
  }

  @Override
  public Vector like(int cardinality) {
    DenseVector denseVector = new DenseVector(cardinality);
    denseVector.setLabelBindings(getLabelBindings());
    return denseVector;
  }

  @Override
  public void setQuick(int index, double value) {
    values[index] = value;
  }

  @Override
  public int getNumNondefaultElements() {
    return values.length;
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > values.length)
      throw new CardinalityException();
    if (offset < 0 || offset + length > values.length)
      throw new IndexException();
    return new VectorView(this, offset, length);
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof DenseVector)
      return other == this;
    else
      return other.haveSharedCells(this);
  }

  /**
   * Returns an iterator that traverses this Vector from 0 to cardinality-1, in
   * that order.
   * 
   * @see java.lang.Iterable#iterator
   */
  @Override
  public java.util.Iterator<Vector.Element> iterator() {
    return new Iterator();
  }

  private class Iterator implements java.util.Iterator<Vector.Element> {
    private int ind;

    private Iterator() {
      ind = 0;
    }

    @Override
    public boolean hasNext() {
      return ind < values.length;
    }

    @Override
    public Vector.Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      return new Element(ind++);
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }


  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(size());
    for (Vector.Element element : this) {
      dataOutput.writeDouble(element.get());
    }
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    double[] values = new double[dataInput.readInt()];
    for (int i = 0; i < values.length; i++) {
      values[i] = dataInput.readDouble();
    }
    this.values = values;
  }

  /**
   * Indicate whether the two objects are the same or not.  Two {@link org.apache.mahout.matrix.Vector}s can be equal
   * even if the underlying implementation is not equal.
   * @param o The object to compare
   * @return true if the objects have the same cell values and same name, false otherwise.
   *
   * @see AbstractVector#strictEquivalence(Vector, Vector)
   * @see AbstractVector#equivalent(Vector, Vector)
   */
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Vector)) return false;

    Vector that = (Vector) o;
    String thatName = that.getName();
    if (this.size() != that.size()) return false;
    if (name != null && thatName != null && !name.equals(thatName)){
      return false;
    } else if ((name != null && thatName == null) || (thatName != null && name == null)){
      return false;
    }

    if (that instanceof DenseVector) {
      if (!Arrays.equals(values, ((DenseVector) that).values)) return false;
    } else {
      return equivalent(this, that);
    }

    return true;
  }

  @Override
  public int hashCode() {
    int result = (values != null ? values.hashCode() : 0);
    result = 31 * result + name.hashCode();

    return result;
  }
}
