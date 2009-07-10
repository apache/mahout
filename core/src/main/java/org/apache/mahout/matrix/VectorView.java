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

  @Override
  public double getQuick(int index) {
    return vector.getQuick(offset + index);
  }

  @Override
  public Vector like() {
    return vector.like();
  }

  @Override
  public Vector like(int cardinality) {
    return vector.like(cardinality);
  }

  @Override
  public void setQuick(int index, double value) {
    vector.setQuick(offset + index, value);
  }

  @Override
  public int getNumNondefaultElements() {
    return cardinality;
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (length > cardinality) {
      {
        throw new CardinalityException();
      }
    }
    if (offset < 0 || offset + length > cardinality) {
      {
        throw new IndexException();
      }
    }
    return new VectorView(vector, offset + this.offset, length);
  }

  @Override
  public boolean haveSharedCells(Vector other) {
    if (other instanceof VectorView) {
      {
        return other == this || vector.haveSharedCells(other);
      }
    } else {
      {
        return other.haveSharedCells(vector);
      }
    }
  }

  /** @return true if index is a valid index in the underlying Vector */
  private boolean isInView(int index) {
    return index >= offset && index < offset + cardinality;
  }

  @Override
  public java.util.Iterator<Vector.Element> iterateNonZero() {
    return new NonZeroIterator();
  }

  @Override
  public java.util.Iterator<Vector.Element> iterateAll() {
    return new AllIterator();
  }

  public class NonZeroIterator implements java.util.Iterator<Vector.Element> {
    private final java.util.Iterator<Vector.Element> it;

    private Vector.Element el;

    public NonZeroIterator() {
      it = vector.iterateAll();
      buffer();
    }

    private void buffer() {
      while (it.hasNext()) {
        el = it.next();
        if (isInView(el.index()) && el.get() != 0) {
          final Vector.Element decorated = vector.getElement(el.index());
          el = new Vector.Element() {
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
              el.set(value);
            }
          };
          return;
        }
      }
      el = null; // No element was found
    }

    @Override
    public Vector.Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Vector.Element buffer = el;
      buffer();
      return buffer;
    }

    @Override
    public boolean hasNext() {
      return el != null;
    }

    /** @throws UnsupportedOperationException all the time. method not implemented. */
    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  public class AllIterator implements java.util.Iterator<Vector.Element> {
    private final java.util.Iterator<Vector.Element> it;

    private Vector.Element el;

    public AllIterator() {
      it = vector.iterateAll();
      buffer();
    }

    private void buffer() {
      while (it.hasNext()) {
        el = it.next();
        if (isInView(el.index())) {
          final Vector.Element decorated = vector.getElement(el.index());
          el = new Vector.Element() {
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
              el.set(value);
            }
          };
          return;
        }
      }
      el = null; // No element was found
    }

    @Override
    public Vector.Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Vector.Element buffer = el;
      buffer();
      return buffer;
    }

    @Override
    public boolean hasNext() {
      return el != null;
    }

    /** @throws UnsupportedOperationException all the time. method not implemented. */
    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }


  @Override
  public double dot(Vector x) {
    if (size() != x.size()) {
      {
        throw new CardinalityException();
      }
    }
    double result = 0;
    for (int i = 0; i < size(); i++) {
      {
        result += getQuick(i) * x.getQuick(i);
      }
    }
    return result;
  }

  @Override
  public Vector.Element getElement(int index) {
    return new Element(index);
  }

  public class Element implements Vector.Element {
    private final int ind;

    public Element(int ind) {
      this.ind = ind;
    }

    @Override
    public double get() {
      return getQuick(ind);
    }

    @Override
    public int index() {
      return ind;
    }

    @Override
    public void set(double value) {
      setQuick(ind, value);
    }
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeUTF(this.name == null ? "" : this.name);
    dataOutput.writeInt(offset);
    dataOutput.writeInt(cardinality);
    writeVector(dataOutput, vector);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    this.name = dataInput.readUTF();
    this.offset = dataInput.readInt();
    this.cardinality = dataInput.readInt();
    this.vector = readVector(dataInput);
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
