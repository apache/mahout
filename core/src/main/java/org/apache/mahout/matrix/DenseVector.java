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

  private double[] values;

  /**
   * Decode a new instance from the argument
   * 
   * @param writableComparable
   *            a WritableComparable<?> produced by the asWritableComparable<?> method
   * @return a DenseVector
   */
  public static Vector decodeFormat(WritableComparable<?> writableComparable) {
    return decodeFormat(writableComparable.toString());
  }

  /**
   * Decode a new instance from the formatted string
   * 
   * @param formattedString
   *            a String produced by asFormatString()
   * @return a DenseVector
   */
  public static Vector decodeFormat(String formattedString) {
    String[] pts = formattedString.split(",");
    double[] point = new double[pts.length - 2];
    for (int i = 1; i < pts.length - 1; i++)
      point[i - 1] = Double.parseDouble(pts[i]);
    return new DenseVector(point);
  }

  /**
   * Construct a new instance using provided values
   * 
   * @param values
   */
  public DenseVector(double[] values) {
    this.values = values.clone();
  }

  /**
   * Construct a new instance of the given cardinality
   * 
   * @param cardinality
   */
  public DenseVector(int cardinality) {
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
  public String asFormatString() {
    StringBuilder out = new StringBuilder();
    out.append("[, ");
    for (double value : values) {
      out.append(value).append(", ");
    }
    out.append("] ");
    return out.toString();
  }

  @Override
  public int cardinality() {
    return values.length;
  }

  @Override
  public DenseVector copy() {
    return new DenseVector(values);
  }

  @Override
  public double getQuick(int index) {
    return values[index];
  }

  @Override
  public DenseVector like() {
    return new DenseVector(cardinality());
  }

  @Override
  public Vector like(int cardinality) {
    return new DenseVector(cardinality);
  }

  @Override
  public void setQuick(int index, double value) {
    values[index] = value;
  }

  @Override
  public int size() {
    return values.length;
  }

  @Override
  public double[] toArray() {
    return values.clone();
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
    dataOutput.writeInt(cardinality());
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

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Vector)) return false;

    Vector that = (Vector) o;
    if (this.cardinality() != that.cardinality()) return false;

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
    result = 31 * result + values.length;
    return result;
  }
}
