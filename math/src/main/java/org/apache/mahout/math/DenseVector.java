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

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

import com.google.common.base.Preconditions;

/** Implements vector as an array of doubles */
public class DenseVector extends AbstractVector {

  private double[] values;

  /** For serialization purposes only */
  public DenseVector() {
    super(0);
  }

  /** Construct a new instance using provided values
   *  @param values - array of values
   */
  public DenseVector(double[] values) {
    this(values, false);
  }

  public DenseVector(double[] values, boolean shallowCopy) {
    super(values.length);
    this.values = shallowCopy ? values : values.clone();
  }

  public DenseVector(DenseVector values, boolean shallowCopy) {
    this(values.values, shallowCopy);
  }

  /** Construct a new instance of the given cardinality
   * @param cardinality - number of values in the vector
   */
  public DenseVector(int cardinality) {
    super(cardinality);
    this.values = new double[cardinality];
  }

  /**
   * Copy-constructor (for use in turning a sparse vector into a dense one, for example)
   * @param vector The vector to copy
   */
  public DenseVector(Vector vector) {
    super(vector.size());
    values = new double[vector.size()];
    for (Element e : vector.nonZeroes()) {
      values[e.index()] = e.get();
    }
  }

  @Override
  public double dot(Vector x) {
    if (!x.isDense()) {
      return super.dot(x);
    } else {

      int size = x.size();
      if (values.length != size) {
        throw new CardinalityException(values.length, size);
      }

      double sum = 0;
      for (int n = 0; n < size; n++) {
        sum += values[n] * x.getQuick(n);
      }
      return sum;
    }
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public DenseVector clone() {
    return new DenseVector(values.clone());
  }

  /**
   * @return true
   */
  @Override
  public boolean isDense() {
    return true;
  }

  /**
   * @return true
   */
  @Override
  public boolean isSequentialAccess() {
    return true;
  }

  @Override
  protected double dotSelf() {
    double result = 0.0;
    int max = size();
    for (int i = 0; i < max; i++) {
      result += values[i] * values[i];
    }
    return result;
  }

  @Override
  public double getQuick(int index) {
    return values[index];
  }

  @Override
  public DenseVector like() {
    return new DenseVector(size());
  }

  @Override
  public void setQuick(int index, double value) {
    invalidateCachedLength();
    values[index] = value;
  }

  @Override
  public void incrementQuick(int index, double increment) {
    invalidateCachedLength();
    values[index] += increment;
  }

  @Override
  public Vector assign(double value) {
    invalidateCachedLength();
    Arrays.fill(values, value);
    return this;
  }

  @Override
  public int getNumNondefaultElements() {
    return values.length;
  }

  public Vector assign(DenseVector vector) {
    // make sure the data field has the correct length
    if (vector.values.length != this.values.length) {
      this.values = new double[vector.values.length];
    }
    // now copy the values
    System.arraycopy(vector.values, 0, this.values, 0, this.values.length);
    return this;
  }

  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    int numUpdates = updates.getNumMappings();
    int[] indices = updates.getIndices();
    double[] values = updates.getValues();
    for (int i = 0; i < numUpdates; ++i) {
      this.values[indices[i]] = values[i];
    }
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (offset < 0) {
      throw new IndexException(offset, size());
    }
    if (offset + length > size()) {
      throw new IndexException(offset + length, size());
    }
    return new VectorView(this, offset, length);
  }

  @Override
  public double getLookupCost() {
    return 1;
  }

  @Override
  public double getIteratorAdvanceCost() {
    return 1;
  }

  @Override
  public boolean isAddConstantTime() {
    return true;
  }

  /**
   * Returns an iterator that traverses this Vector from 0 to cardinality-1, in that order.
   */
  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonDefaultIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof DenseVector) {
      // Speedup for DenseVectors
      return Arrays.equals(values, ((DenseVector) o).values);
    }
    return super.equals(o);
  }

  public void addAll(Vector v) {
    if (size() != v.size()) {
      throw new CardinalityException(size(), v.size());
    }

    for (Element element : v.nonZeroes()) {
      values[element.index()] += element.get();
    }
  }

  private final class NonDefaultIterator implements Iterator<Element> {
    private final DenseElement element = new DenseElement();
    private int index = -1;
    private int lookAheadIndex = -1;

    @Override
    public boolean hasNext() {
      if (lookAheadIndex == index) {  // User calls hasNext() after a next()
        lookAhead();
      } // else user called hasNext() repeatedly.
      return lookAheadIndex < size();
    }

    private void lookAhead() {
      lookAheadIndex++;
      while (lookAheadIndex < size() && values[lookAheadIndex] == 0.0) {
        lookAheadIndex++;
      }
    }

    @Override
    public Element next() {
      if (lookAheadIndex == index) { // If user called next() without checking hasNext().
        lookAhead();
      }

      Preconditions.checkState(lookAheadIndex > index);
      index = lookAheadIndex;

      if (index >= size()) { // If the end is reached.
        throw new NoSuchElementException();
      }

      element.index = index;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class AllIterator implements Iterator<Element> {
    private final DenseElement element = new DenseElement();

    private AllIterator() {
      element.index = -1;
    }

    @Override
    public boolean hasNext() {
      return element.index + 1 < size();
    }

    @Override
    public Element next() {
      if (element.index + 1 >= size()) { // If the end is reached.
        throw new NoSuchElementException();
      }
      element.index++;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class DenseElement implements Element {
    int index;

    @Override
    public double get() {
      return values[index];
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      invalidateCachedLength();
      values[index] = value;
    }
  }
}
