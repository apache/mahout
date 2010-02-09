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

/** Implements subset view of a Vector */
public class VectorView extends AbstractVector {

  private Vector vector;

  // the offset into the Vector
  private int offset;

  // the cardinality of the view
  private int cardinality;

  /** For serialization purposes only */
  public VectorView() {
  }

  public VectorView(Vector vector, int offset, int cardinality) {
    this.vector = vector;
    this.offset = offset;
    this.cardinality = cardinality;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return ((AbstractVector) vector).matrixLike(rows, columns);
  }

  @Override
  public int size() {
    return cardinality;
  }

  @Override
  public Vector clone() {
    VectorView clone = (VectorView) super.clone();
    clone.vector = vector.clone();
    return clone;
  }

  public double getQuick(int index) {
    return vector.getQuick(offset + index);
  }

  public Vector like() {
    return vector.like();
  }

  public Vector like(int cardinality) {
    return vector.like(cardinality);
  }

  public void setQuick(int index, double value) {
    vector.setQuick(offset + index, value);
  }

  public int getNumNondefaultElements() {
    return cardinality;
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > cardinality) {
      throw new CardinalityException();
    }
    if (offset < 0 || offset + length > cardinality) {
      throw new IndexException();
    }
    return new VectorView(vector, offset + this.offset, length);
  }

  /** @return true if index is a valid index in the underlying Vector */
  private boolean isInView(int index) {
    return index >= offset && index < offset + cardinality;
  }

  public Iterator<Vector.Element> iterateNonZero() {
    return new NonZeroIterator();
  }

  public Iterator<Vector.Element> iterateAll() {
    return new AllIterator();
  }

  public class NonZeroIterator implements Iterator<Vector.Element> {

    private final Iterator<Vector.Element> it;

    private Vector.Element el;

    private NonZeroIterator() {
      it = vector.iterateAll();
      buffer();
    }

    private void buffer() {
      while (it.hasNext()) {
        el = it.next();
        if (isInView(el.index()) && el.get() != 0) {
          final Vector.Element decorated = vector.getElement(el.index());
          el = new Vector.Element() {
            public double get() {
              return decorated.get();
            }

            public int index() {
              return decorated.index() - offset;
            }

            public void set(double value) {
              decorated.set(value);
            }
          };
          return;
        }
      }
      el = null; // No element was found
    }

    public Vector.Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Vector.Element buffer = el;
      buffer();
      return buffer;
    }

    public boolean hasNext() {
      return el != null;
    }

    /** @throws UnsupportedOperationException all the time. method not implemented. */
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  public class AllIterator implements Iterator<Vector.Element> {

    private final Iterator<Vector.Element> it;

    private Vector.Element el;

    private AllIterator() {
      it = vector.iterateAll();
      buffer();
    }

    private void buffer() {
      while (it.hasNext()) {
        el = it.next();
        if (isInView(el.index())) {
          final Vector.Element decorated = vector.getElement(el.index());
          el = new Vector.Element() {
            public double get() {
              return decorated.get();
            }

            public int index() {
              return decorated.index() - offset;
            }

            public void set(double value) {
              decorated.set(value);
            }
          };
          return;
        }
      }
      el = null; // No element was found
    }

    public Vector.Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Vector.Element buffer = el;
      buffer();
      return buffer;
    }

    public boolean hasNext() {
      return el != null;
    }

    /** @throws UnsupportedOperationException all the time. method not implemented. */
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }


  @Override
  public double dot(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    double result = 0;
    for (int i = 0; i < size(); i++) {
      result += getQuick(i) * x.getQuick(i);
    }
    return result;
  }

  public Vector.Element getElement(int index) {
    return new Element(index);
  }

  public class Element implements Vector.Element {

    private final int ind;

    private Element(int ind) {
      this.ind = ind;
    }

    public double get() {
      return getQuick(ind);
    }

    public int index() {
      return ind;
    }

    public void set(double value) {
      setQuick(ind, value);
    }
  }

  @Override
  public boolean equals(Object o) {
    return this == o || (o instanceof Vector && equivalent(this, (Vector) o));

  }

  @Override
  public int hashCode() {
    int result = vector.hashCode();
    result = 31 * result + offset;
    result = 31 * result + cardinality;
    return result;
  }

  @Override
  public double getLengthSquared() {
    double result = 0.0;
    for (int i = 0; i < cardinality; i++) {
      double value = getQuick(i);
      result += value * value;
    }
    return result;
  }

  @Override
  public double getDistanceSquared(Vector v) {
    double result = 0.0;
    for (int i = 0; i < cardinality; i++) {
      double delta = getQuick(i) - v.getQuick(i);
      result += delta * delta;
    }
    return result;
  }

  @Override
  public void addTo(Vector v) {
    Iterator<Vector.Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      v.set(elt.index(), elt.get() + v.get(elt.index()));
    }
  }

}
