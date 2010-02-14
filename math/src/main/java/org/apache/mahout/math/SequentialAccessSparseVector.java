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

/**
 * <p>
 * Implements vector that only stores non-zero doubles as a pair of parallel arrays (OrderedIntDoubleMapping),
 * one int[], one double[].  If there are <b>k</b> non-zero elements in the vector, this implementation has
 * O(log(k)) random-access read performance, and O(k) random-access write performance, which is far below that
 * of the hashmap based {@link org.apache.mahout.math.RandomAccessSparseVector RandomAccessSparseVector}.  This
 * class is primarily used for operations where the all the elements will be accessed in a read-only fashion
 * sequentially: methods which operate not via get() or set(), but via iterateNonZero(), such as (but not limited
 * to) :</p>
 * <ul>
 *   <li>dot(Vector)</li>
 *   <li>addTo(Vector)</li>
 * </ul>
 * <p>
 * Note that the Vector passed to these above methods may (and currently, are) be used in a random access fashion,
 * so for example, calling SequentialAccessSparseVector.dot(SequentialAccessSparseVector) is slow.
 * TODO: this need not be the case - both are ordered, so this should be very fast if implmented in this class
 * </p>
 *
 * {@see OrderedIntDoubleMapping}
 */
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

  public SequentialAccessSparseVector(SequentialAccessSparseVector other, boolean shallowCopy) {
    super(other.getName(), other.size());
    values = shallowCopy ? other.values : other.values.clone();
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

  public double getQuick(int index) {
    return values.get(index);
  }

  public void setQuick(int index, double value) {
    values.set(index, value);
  }

  public int getNumNondefaultElements() {
    return values.getNumMappings();
  }

  public SequentialAccessSparseVector like() {
    int numValues = 256;
    if (values != null) {
      numValues = values.getNumMappings();
    }
    return new SequentialAccessSparseVector(size(), numValues);
  }

  public Vector like(int newCardinality) {
    int numValues = 256;
    if (values != null) {
      numValues = values.getNumMappings();
    }
    return new SequentialAccessSparseVector(newCardinality, numValues);
  }

  public Iterator<Element> iterateNonZero() {
    return new IntDoublePairIterator(this);
  }

  public Iterator<Element> iterateAll() {
    return new IntDoublePairIterator(this, size());
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

    IntDoublePairIterator(SequentialAccessSparseVector v) {
      element = new SparseElement(offset, v);
      maxOffset = v.values.getNumMappings();
    }
    IntDoublePairIterator(SequentialAccessSparseVector v, int cardinality) {
      element = new DenseElement(offset, v);
      maxOffset = cardinality;
    }

    public boolean hasNext() {
      return offset < maxOffset;
    }

    public Element next() {
      if (offset >= maxOffset) {
        throw new NoSuchElementException();
      }
      element.offset = offset++;
      return element;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  public Element getElement(int index) {
    return new DenseElement(index, this);
  }


  private abstract static class AbstractElement implements Element {
    int offset;
    final OrderedIntDoubleMapping mapping;
    final int[] indices;
    final double[] values;

    AbstractElement(int ind, SequentialAccessSparseVector v) {
      offset = ind;
      mapping = v.values;
      values = mapping.getValues();
      indices = mapping.getIndices();
    }
  }

  private static final class DenseElement extends AbstractElement {
    int index;
    SequentialAccessSparseVector v;
    DenseElement(int ind, SequentialAccessSparseVector v) {
      super(ind, v);
      this.v = v;
      index = ind;
    }

    public double get() {
      if(index >= indices.length) return 0.0;
      int cur = indices[index];
      while(cur < offset && index < indices.length - 1) cur = indices[++index];
      if(cur == offset) return values[index];
      return 0.0;
    }

    public int index() {
      return offset;
    }

    public void set(double value) {
      v.lengthSquared = -1;
      if(value != 0.0) mapping.set(indices[offset], value);
    }
  }

  private static final class SparseElement extends AbstractElement {

    SparseElement(int ind, SequentialAccessSparseVector v) {
      super(ind, v);
    }

    public double get() {
      return values[offset];
    }

    public int index() {
      return indices[offset];
    }

    public void set(double value) {
      values[offset] = value;
    }
  }
}
