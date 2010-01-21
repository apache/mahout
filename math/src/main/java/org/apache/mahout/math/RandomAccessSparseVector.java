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

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;


/** Implements vector that only stores non-zero doubles */
public class RandomAccessSparseVector extends AbstractVector {

  protected OpenIntDoubleHashMap values;

  /** For serialization purposes only. */
  public RandomAccessSparseVector() {
  }

  public RandomAccessSparseVector(int cardinality) {
    this(null, cardinality, cardinality / 8); // arbitrary estimate of
    // 'sparseness'
  }

  public RandomAccessSparseVector(int cardinality, int size) {
    this(null, cardinality, size);
  }

  public RandomAccessSparseVector(String name, int cardinality) {
    this(name, cardinality, cardinality / 8); // arbitrary estimate of
    // 'sparseness'
  }

  public RandomAccessSparseVector(String name, int cardinality, int size) {
    super(name, cardinality);
    values = new OpenIntDoubleHashMap(size);
  }

  public RandomAccessSparseVector(Vector other) {
    this(other.getName(), other.size(), other.getNumNondefaultElements());
    Iterator<Vector.Element> it = other.iterateNonZero();
    Vector.Element e;
    while(it.hasNext() && (e = it.next()) != null) {
      values.put(e.index(), e.get());
    }
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    int[] cardinality = {rows, columns};
    return new SparseRowMatrix(cardinality);
  }

  @Override
  public RandomAccessSparseVector clone() {
    RandomAccessSparseVector clone = (RandomAccessSparseVector) super.clone();
    clone.values = (OpenIntDoubleHashMap)values.clone();
    return clone;
  }

  @Override
  public Vector assign(Vector other) {
    if (other.size() != size()) {
      throw new CardinalityException();
    }
    values.clear();
    Iterator<Vector.Element> it = other.iterateNonZero();
    Vector.Element e;
    while(it.hasNext() && (e = it.next()) != null) {
      setQuick(e.index(), e.get());
    }
    return this;
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
  public RandomAccessSparseVector like() {
    int numValues = 256;
    if (values != null) {
      numValues = values.size();
    }
    return new RandomAccessSparseVector(size(), numValues);
  }

  @Override
  public Vector like(int newCardinality) {
    int numValues = 256;
    if (values != null) {
      numValues = values.size();
    }
    return new RandomAccessSparseVector(newCardinality, numValues);
  }

  /**
   * NOTE: this implementation reuses the Vector.Element instance for each call of next(). If you need to preserve the
   * instance, you need to make a copy of it
   *
   * @return an {@link NonZeroIterator} over the Elements.
   * @see #getElement(int)
   */
  @Override
  public java.util.Iterator<Vector.Element> iterateNonZero() {
    return new NonZeroIterator(false);
  }
  
  @Override
  public Iterator<Vector.Element> iterateAll() {
    return new AllIterator();
  }

  /**
   * Indicate whether the two objects are the same or not. Two {@link org.apache.mahout.math.Vector}s can be equal
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

    return equivalent(this, that);
  }

  private class AllIterator implements java.util.Iterator<Vector.Element> {
    private int offset = 0;
    private final Element element = new Element(0);

    @Override
    public boolean hasNext() {
      return offset < size();
    }

    @Override
    public Vector.Element next() {
      if (offset >= size()) {
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


  private class NonZeroIterator implements java.util.Iterator<Vector.Element> {
    private int offset = 0;
    private final Element element = new Element(0);

    private final IntArrayList intArrList =  values.keys();
    
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

  private class DistanceSquared implements IntDoubleProcedure {
    final Vector v;
    public double result = 0.0;

    DistanceSquared(Vector v) {
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
    final Vector v;

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
    if (v.size() != size()) {
      throw new CardinalityException();
    }
    values.forEachPair(new AddToVector(v));
  }

}
