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

/** Implements vector that only stores non-zero doubles */
public class SequentialAccessSparseVector extends AbstractVector {

  protected OrderedIntDoubleMapping values;


  /** For serialization purposes only. */
  public SequentialAccessSparseVector() {
    super(null, 0);
  }

  public SequentialAccessSparseVector(int cardinality, int size) {
    this(null, cardinality, size);
  }

  public SequentialAccessSparseVector(String name, int cardinality, int size) {
    super(name, cardinality);
    values = new OrderedIntDoubleMapping(size);
  }

  public SequentialAccessSparseVector(String name, int cardinality) {
    this(name, cardinality, cardinality / 8); // arbitrary estimate of
    // 'sparseness'
  }

  public SequentialAccessSparseVector(int cardinality) {
    this(null, cardinality, cardinality / 8); // arbitrary estimate of
    // 'sparseness'
  }

  public SequentialAccessSparseVector(Vector other) {
    this(other.getName(), other.size(), other.getNumNondefaultElements());
    Iterator<Element> it = other.iterateNonZero();
    Element e;
    while(it.hasNext() && (e = it.next()) != null) {
      set(e.index(), e.get());
    }
  }

  public SequentialAccessSparseVector(SequentialAccessSparseVector other) {
    this(other.getName(), other.size(), other.getNumNondefaultElements());
    values = other.values.clone();
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    int[] cardinality = {rows, columns};
    return new SparseRowMatrix(cardinality);
  }

  @Override
  public SequentialAccessSparseVector clone() {
    SequentialAccessSparseVector clone = (SequentialAccessSparseVector) super.clone();
    clone.values = values.clone();
    return clone;
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

  private static class DistanceSquarer implements Iterator<Element> {
    final Iterator<Element> it1;
    final Iterator<Element> it2;
    Element e1 = null;
    Element e2 = null;
    boolean firstIteration = true;
    Iterator<Element> notExhausted = null;
    double v = 0;

    DistanceSquarer(Iterator<Element> it1, Iterator<Element> it2) {
      this.it1 = it1;
      this.it2 = it2;
    }

    @Override
    public boolean hasNext() {
      return firstIteration || it1.hasNext() || it2.hasNext();
    }

    @Override
    public Element next() {
      if(firstIteration) {
        e1 = it1.next();
        e2 = it2.next();
        firstIteration = false;
      }
      Iterator<Element> it;
      Element e = null;
      if(notExhausted != null) {
        if(notExhausted.hasNext()) {
          e = notExhausted.next();
          v += e.get() * e.get();
        }
      } else if(e1.index() < e2.index()) {
        v += e1.get() * e1.get();
        e = e1;
        if(it1.hasNext()) {
          e1 = it1.next();
        } else {
          notExhausted = it2;
        }
      } else if(e1.index() > e2.index()) {
        v += e2.get() * e2.get();
        e = e2;
        if(it2.hasNext()) {
          e2 = it2.next();
        } else {
          notExhausted = it1;
        }
      } else {
        double d = e1.get() - e2.get();
        if(it1.hasNext()) {
          e1 = it1.next();
          e = e1;
        } else if(it2.hasNext()) {
          e2 = it2.next();
          e = e2;
        } else {
          e = null;
        }
        v += d*d;
      }
      return e;
    }

    @Override
    public void remove() {
      // ignore
    }

    public double distanceSquared() {
      return v;
    }
  }

  @Override
  public double getDistanceSquared(Vector v) {
    if(v instanceof SequentialAccessSparseVector) {
      Iterator<Element> it = iterateNonZero();
      Iterator<Element> vIt = v.iterateNonZero();
      if(!it.hasNext()) {
        return v.dot(v);
      }
      if(!vIt.hasNext()) {
        return this.dot(this);
      }
      DistanceSquarer d = new DistanceSquarer(it, vIt);
      while(d.hasNext()) {
        d.next();
      }
      return d.distanceSquared();
    }
    else {
      return super.getDistanceSquared(v);
    }
  }

  @Override
  public SequentialAccessSparseVector like() {
    int numValues = 256;
    if (values != null) {
      numValues = values.getNumMappings();
    }
    return new SequentialAccessSparseVector(size(), numValues);
  }

  @Override
  public Vector like(int newCardinality) {
    int numValues = 256;
    if (values != null) {
      numValues = values.getNumMappings();
    }
    return new SequentialAccessSparseVector(newCardinality, numValues);
  }

  @Override
  public java.util.Iterator<Element> iterateNonZero() {
    return new IntDoublePairIterator(values);
  }

  @Override
  public Iterator<Element> iterateAll() {
    return new IntDoublePairIterator(values, size());
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

    if (that instanceof SequentialAccessSparseVector) {
      return (values == null ? ((SequentialAccessSparseVector) that).values == null : values
          .equals(((SequentialAccessSparseVector) that).values));
    } else {
      return equivalent(this, that);
    }

  }

  private static final class IntDoublePairIterator implements java.util.Iterator<Element> {
    private int offset = 0;
    private final AbstractElement element;
    private final int maxOffset;

    IntDoublePairIterator(OrderedIntDoubleMapping mapping) {
      element = new SparseElement(offset, mapping);
      maxOffset = mapping.getNumMappings();
    }
    IntDoublePairIterator(OrderedIntDoubleMapping mapping, int cardinality) {
      element = new DenseElement(offset, mapping);
      maxOffset = cardinality;
    }
    @Override
    public boolean hasNext() {
      return offset < maxOffset;
    }

    @Override
    public Element next() {
      if (offset >= maxOffset) {
        throw new NoSuchElementException();
      }
      element.offset = offset++;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public Element getElement(int index) {
    return new DenseElement(index, values);
  }


  private abstract static class AbstractElement implements Element {
    int offset;
    final OrderedIntDoubleMapping mapping;
    final int[] indices;
    final double[] values;

    AbstractElement(int ind, OrderedIntDoubleMapping m) {
      offset = ind;
      mapping = m;
      values = m.getValues();
      indices = m.getIndices();
    }
  }

  private static final class DenseElement extends AbstractElement {
    int index;

    DenseElement(int ind, OrderedIntDoubleMapping mapping) {
      super(ind, mapping);
      index = ind;
    }

    @Override
    public double get() {
      if(index >= indices.length) return 0.0;
      int cur = indices[index];
      while(cur < offset && index < indices.length - 1) cur = indices[++index];
      if(cur == offset) return values[index];
      return 0.0;
    }

    @Override
    public int index() {
      return offset;
    }

    @Override
    public void set(double value) {
      if(value != 0.0) mapping.set(indices[offset], value);
    }
  }

  private static final class SparseElement extends AbstractElement {

    SparseElement(int ind, OrderedIntDoubleMapping mapping) {
      super(ind, mapping);
    }

    @Override
    public double get() {
      return values[offset];
    }

    @Override
    public int index() {
      return indices[offset];
    }

    @Override
    public void set(double value) {
      values[offset] = value;
    }
  }
}
