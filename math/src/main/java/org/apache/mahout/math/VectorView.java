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

import com.google.common.collect.AbstractIterator;

/** Implements subset view of a Vector */
public class VectorView extends AbstractVector {

  private Vector vector;

  // the offset into the Vector
  private int offset;

  /** For serialization purposes only */
  public VectorView() {
    super(0);
  }

  public VectorView(Vector vector, int offset, int cardinality) {
    super(cardinality);
    this.vector = vector;
    this.offset = offset;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return ((AbstractVector) vector).matrixLike(rows, columns);
  }

  @Override
  public Vector clone() {
    VectorView r = (VectorView) super.clone();
    r.vector = vector.clone();
    r.offset = offset;
    return r;
  }

  @Override
  public boolean isDense() {
    return vector.isDense();
  }

  @Override
  public boolean isSequentialAccess() {
    return vector.isSequentialAccess();
  }

  @Override
  public VectorView like() {
    return new VectorView(vector.like(), offset, size());
  }

  @Override
  public double getQuick(int index) {
    return vector.getQuick(offset + index);
  }

  @Override
  public void setQuick(int index, double value) {
    vector.setQuick(offset + index, value);
  }

  @Override
  public int getNumNondefaultElements() {
    return size();
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (offset < 0) {
      throw new IndexException(offset, size());
    }
    if (offset + length > size()) {
      throw new IndexException(offset + length, size());
    }
    return new VectorView(vector, offset + this.offset, length);
  }

  /** @return true if index is a valid index in the underlying Vector */
  private boolean isInView(int index) {
    return index >= offset && index < offset + size();
  }

  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonZeroIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  public final class NonZeroIterator extends AbstractIterator<Element> {

    private final Iterator<Element> it;

    private NonZeroIterator() {
      it = vector.nonZeroes().iterator();
    }

    @Override
    protected Element computeNext() {
      while (it.hasNext()) {
        Element el = it.next();
        if (isInView(el.index()) && el.get() != 0) {
          Element decorated = vector.getElement(el.index());
          return new DecoratorElement(decorated);
        }
      }
      return endOfData();
    }

  }

  public final class AllIterator extends AbstractIterator<Element> {

    private final Iterator<Element> it;

    private AllIterator() {
      it = vector.all().iterator();
    }

    @Override
    protected Element computeNext() {
      while (it.hasNext()) {
        Element el = it.next();
        if (isInView(el.index())) {
          Element decorated = vector.getElement(el.index());
          return new DecoratorElement(decorated);
        }
      }
      return endOfData(); // No element was found
    }

  }

  private final class DecoratorElement implements Element {

    private final Element decorated;

    private DecoratorElement(Element decorated) {
      this.decorated = decorated;
    }

    @Override
    public double get() {
      return decorated.get();
    }

    @Override
    public int index() {
      return decorated.index() - offset;
    }

    @Override
    public void set(double value) {
      decorated.set(value);
    }
  }

  @Override
  public double getLengthSquared() {
    double result = 0.0;
    int size = size();
    for (int i = 0; i < size; i++) {
      double value = getQuick(i);
      result += value * value;
    }
    return result;
  }

  @Override
  public double getDistanceSquared(Vector v) {
    double result = 0.0;
    int size = size();
    for (int i = 0; i < size; i++) {
      double delta = getQuick(i) - v.getQuick(i);
      result += delta * delta;
    }
    return result;
  }

  @Override
  public double getLookupCost() {
    return vector.getLookupCost();
  }

  @Override
  public double getIteratorAdvanceCost() {
    // TODO: remove the 2x after fixing the Element iterator
    return 2 * vector.getIteratorAdvanceCost();
  }

  @Override
  public boolean isAddConstantTime() {
    return vector.isAddConstantTime();
  }

  /**
   * Used internally by assign() to update multiple indices and values at once.
   * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
   * <p/>
   * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
   *
   * @param updates a mapping of indices to values to merge in the vector.
   */
  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    for (int i = 0; i < updates.getNumMappings(); ++i) {
      updates.setIndexAt(i, updates.indexAt(i) + offset);
    }
    vector.mergeUpdates(updates);
  }
}
