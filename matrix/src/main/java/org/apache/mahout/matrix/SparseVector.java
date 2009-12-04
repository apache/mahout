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

import org.apache.mahout.matrix.function.IntDoubleProcedure;
import org.apache.mahout.matrix.list.DoubleArrayList;
import org.apache.mahout.matrix.list.IntArrayList;
import org.apache.mahout.matrix.map.OpenIntDoubleHashMap;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;
import java.util.NoSuchElementException;


/** Implements vector that only stores non-zero doubles */
public class SparseVector extends AbstractVector {

  private OpenIntDoubleHashMap values;

  private int cardinality;
  private double lengthSquared = -1.0;

  /** For serialization purposes only. */
  public SparseVector() {
  }

  public SparseVector(int cardinality, int size) {
    this(null, cardinality, size);
  }

  public SparseVector(String name, int cardinality, int size) {
    super(name);
    values = new OpenIntDoubleHashMap(size);
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
    return new SparseRowMatrix(new int[] {rows, columns});
  }

  @Override
  public int size() {
    return cardinality;
  }

  @Override
  public SparseVector clone() {
    SparseVector clone = (SparseVector) super.clone();
    clone.values = (OpenIntDoubleHashMap) values.clone();
    return clone;
  }

  @Override
  public double getQuick(int index) {
    return values.get(index);
  }

  @Override
  public void setQuick(int index, double value) {
    values.put(index, value);
  }

  @Override
  public int getNumNondefaultElements() {
    return values.size();
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > cardinality) {
      throw new CardinalityException();
    }
    if (offset < 0 || offset + length > cardinality) {
      throw new IndexException();
    }
    return new VectorView(this, offset, length);
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof SparseVector) {
      return other == this;
    } else {
      return other.haveSharedCells(this);
    }
  }

  @Override
  public SparseVector like() {
    int numValues = 256;
    if (values != null) {
      numValues = values.size();
    }
    return new SparseVector(cardinality, numValues);
  }

  @Override
  public Vector like(int newCardinality) {
    int numValues = 256;
    if (values != null) {
      numValues = values.size();
    }
    return new SparseVector(newCardinality, numValues);
  }

  /**
   * NOTE: this implementation reuses the Vector.Element instance for each call of next(). If you need to preserve the
   * instance, you need to make a copy of it
   *
   * @return an {@link NonZeroIterator} over the Elements.
   * @see #getElement(int)
   */
  @Override
  public Iterator<Vector.Element> iterateNonZero() {
    return new NonZeroIterator(false);
  }

  @Override
  public Iterator<Vector.Element> iterateNonZero(boolean sorted) {
    return new NonZeroIterator(sorted);
  }

  @Override
  public Iterator<Vector.Element> iterateAll() {
    return new AllIterator();
  }

  /**
   * Indicate whether the two objects are the same or not. Two {@link org.apache.mahout.matrix.Vector}s can be equal
   * even if the underlying implementation is not equal.
   *
   * @param o The object to compare
   * @return true if the objects have the same cell values and same name, false otherwise. <p/> * @see
   *         AbstractVector#strictEquivalence(Vector, Vector)
   * @see AbstractVector#equivalent(Vector, Vector)
   */
  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Vector)) {
      return false;
    }

    Vector that = (Vector) o;
    String thisName = getName();
    String thatName = that.getName();
    if (this.size() != that.size()) {
      return false;
    }
    if (thisName != null && thatName != null && !thisName.equals(thatName)) {
      return false;
    } else if ((thisName != null && thatName == null)
        || (thatName != null && thisName == null)) {
      return false;
    }

    if (that instanceof SparseVector) {
      return (values == null ? ((SparseVector) that).values == null : values
          .equals(((SparseVector) that).values));
    } else {
      return equivalent(this, that);
    }

  }

  private class AllIterator implements Iterator<Vector.Element> {

    private int offset = 0;
    private final Element element = new Element(0);

    @Override
    public boolean hasNext() {
      return offset < cardinality;
    }

    @Override
    public Vector.Element next() {
      if (offset >= cardinality) {
        throw new NoSuchElementException();
      }
      element.ind = offset++;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }


  private class NonZeroIterator implements Iterator<Vector.Element> {

    private int offset = 0;
    private final Element element = new Element(0);

    private final IntArrayList intArrList = values.keys();

    private NonZeroIterator(boolean sorted) {
      if (sorted) {
        intArrList.sort();
      }
    }

    @Override
    public boolean hasNext() {
      return offset < intArrList.size();
    }

    @Override
    public Element next() {
      if (offset < intArrList.size()) {
        element.ind = intArrList.get(offset++);
        return element;
      }
      throw new NoSuchElementException();
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public Vector.Element getElement(int index) {
    return new Element(index);
  }

  public class Element implements Vector.Element {

    private int ind;

    public Element(int ind) {
      this.ind = ind;
    }

    @Override
    public double get() {
      return values.get(ind);
    }

    @Override
    public int index() {
      return ind;
    }

    @Override
    public void set(double value) {
      values.put(ind, value);
    }
  }


  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(this.getName() == null ? "" : this.getName());
    dataOutput.writeInt(size());
    int nde = getNumNondefaultElements();
    dataOutput.writeInt(nde);
    Iterator<Vector.Element> iter = iterateNonZero();
    int count = 0;
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      dataOutput.writeInt(element.index());
      dataOutput.writeDouble(element.get());
      count++;
    }
    assert (nde == count);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    this.setName(dataInput.readUTF());
    this.cardinality = dataInput.readInt();
    int size = dataInput.readInt();
    OpenIntDoubleHashMap values = new OpenIntDoubleHashMap((int) (size * 1.5));
    int i = 0;
    while (i < size) {
      int index = dataInput.readInt();
      double value = dataInput.readDouble();
      values.put(index, value);
      i++;
    }
    assert (i == size);
    this.values = values;
    this.lengthSquared = -1.0;
  }


  @Override
  public double getLengthSquared() {
    if (lengthSquared >= 0.0) {
      return lengthSquared;
    }
    double result = 0.0;
    DoubleArrayList valueList = values.values();
    for (int i = 0; i < valueList.size(); i++) {
      double val = valueList.get(i);
      result += val * val;
    }
    lengthSquared = result;
    return result;
  }

  private class DistanceSquared implements IntDoubleProcedure {

    private final Vector v;
    private double result = 0.0;

    private DistanceSquared(Vector v) {
      this.v = v;
    }

    @Override
    public boolean apply(int key, double value) {
      double centroidValue = v.get(key);
      double delta = value - centroidValue;
      result += (delta * delta) - (centroidValue * centroidValue);
      return true;
    }
  }

  @Override
  public double getDistanceSquared(Vector v) {
    //TODO: Check sizes?

    DistanceSquared distanceSquared = new DistanceSquared(v);
    values.forEachPair(distanceSquared);
    return distanceSquared.result;
  }

  private class AddToVector implements IntDoubleProcedure {

    private final Vector v;

    private AddToVector(Vector v) {
      this.v = v;
    }

    @Override
    public boolean apply(int key, double value) {
      v.set(key, value + v.get(key));
      return true;
    }
  }

  @Override
  public void addTo(Vector v) {
    values.forEachPair(new AddToVector(v));
  }

}
