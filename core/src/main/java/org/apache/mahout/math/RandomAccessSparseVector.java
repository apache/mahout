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

import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap.Entry;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.mahout.math.set.AbstractSet;

/** Implements vector that only stores non-zero doubles */
public class RandomAccessSparseVector extends AbstractVector {

  private static final int INITIAL_CAPACITY = 11;

  private Int2DoubleOpenHashMap values;

  /** For serialization purposes only. */
  public RandomAccessSparseVector() {
    super(0);
  }

  public RandomAccessSparseVector(int cardinality) {
    this(cardinality, Math.min(cardinality, INITIAL_CAPACITY)); // arbitrary estimate of 'sparseness'
  }

  public RandomAccessSparseVector(int cardinality, int initialCapacity) {
    super(cardinality);
    values = new Int2DoubleOpenHashMap(initialCapacity, .5f);
  }

  public RandomAccessSparseVector(Vector other) {
    this(other.size(), other.getNumNondefaultElements());
    for (Element e : other.nonZeroes()) {
      values.put(e.index(), e.get());
    }
  }

  private RandomAccessSparseVector(int cardinality, Int2DoubleOpenHashMap values) {
    super(cardinality);
    this.values = values;
  }

  public RandomAccessSparseVector(RandomAccessSparseVector other, boolean shallowCopy) {
    super(other.size());
    values = shallowCopy ? other.values : other.values.clone();
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new SparseMatrix(rows, columns);
  }

  @Override
  public RandomAccessSparseVector clone() {
    return new RandomAccessSparseVector(size(), values.clone());
  }

  @Override
  public String toString() {
    return sparseVectorToString();
  }

  @Override
  public Vector assign(Vector other) {
    if (size() != other.size()) {
      throw new CardinalityException(size(), other.size());
    }
    values.clear();
    for (Element e : other.nonZeroes()) {
      setQuick(e.index(), e.get());
    }
    return this;
  }

  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    for (int i = 0; i < updates.getNumMappings(); ++i) {
      values.put(updates.getIndices()[i], updates.getValues()[i]);
    }
  }

  /**
   * @return false
   */
  @Override
  public boolean isDense() {
    return false;
  }

  /**
   * @return false
   */
  @Override
  public boolean isSequentialAccess() {
    return false;
  }

  @Override
  public double getQuick(int index) {
    return values.get(index);
  }

  @Override
  public void setQuick(int index, double value) {
    invalidateCachedLength();
    if (value == 0.0) {
      values.remove(index);
    } else {
      values.put(index, value);
    }
  }

  @Override
  public void incrementQuick(int index, double increment) {
    invalidateCachedLength();
    values.addTo( index, increment);
  }


  @Override
  public RandomAccessSparseVector like() {
    return new RandomAccessSparseVector(size(), values.size());
  }

  @Override
  public Vector like(int cardinality) {
    return new RandomAccessSparseVector(cardinality, values.size());
  }

  @Override
  public int getNumNondefaultElements() {
    return values.size();
  }

  @Override
  public int getNumNonZeroElements() {
    final DoubleIterator iterator = values.values().iterator();
    int numNonZeros = 0;
    for( int i = values.size(); i-- != 0; ) if ( iterator.nextDouble() != 0 ) numNonZeros++;
    return numNonZeros;
  }

  @Override
  public double getLookupCost() {
    return 1;
  }

  @Override
  public double getIteratorAdvanceCost() {
    return 1 + (AbstractSet.DEFAULT_MAX_LOAD_FACTOR + AbstractSet.DEFAULT_MIN_LOAD_FACTOR) / 2;
  }

  /**
   * This is "sort of" constant, but really it might resize the array.
   */
  @Override
  public boolean isAddConstantTime() {
    return true;
  }

  /*
  @Override
  public Element getElement(int index) {
    // TODO: this should return a MapElement so as to avoid hashing for both getQuick and setQuick.
    return super.getElement(index);
  }
   */

  private final class NonZeroIterator implements Iterator<Element> {
    final ObjectIterator<Int2DoubleMap.Entry> fastIterator = values.int2DoubleEntrySet().fastIterator();
    final RandomAccessElement element = new RandomAccessElement( fastIterator );

    @Override
    public boolean hasNext() {
      return fastIterator.hasNext();
    }

    @Override
    public Element next() {
      if ( ! hasNext() ) throw new NoSuchElementException();
      element.entry = fastIterator.next();
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  final class RandomAccessElement implements Element {
    Int2DoubleMap.Entry entry;
    final ObjectIterator<Int2DoubleMap.Entry> fastIterator;

    public RandomAccessElement( ObjectIterator<Entry> fastIterator ) {
      super();
      this.fastIterator = fastIterator;
    }

    @Override
    public double get() {
      return entry.getDoubleValue();
    }

    @Override
    public int index() {
      return entry.getIntKey();
    }

    @Override
    public void set( double value ) {
      invalidateCachedLength();
      if (value == 0.0) fastIterator.remove();
      else entry.setValue( value );
    }
  }
  /**
   * NOTE: this implementation reuses the Vector.Element instance for each call of next(). If you need to preserve the
   * instance, you need to make a copy of it
   *
   * @return an {@link Iterator} over the Elements.
   * @see #getElement(int)
   */
  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonZeroIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  final class GeneralElement implements Element {
    int index;
    double value;

    @Override
    public double get() {
      return value;
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set( double value ) {
      invalidateCachedLength();
      if (value == 0.0) values.remove( index );
      else values.put( index, value );
    }
}

  private final class AllIterator implements Iterator<Element> {
    private final GeneralElement element = new GeneralElement();

    private AllIterator() {
      element.index = -1;
    }

    @Override
    public boolean hasNext() {
      return element.index + 1 < size();
    }

    @Override
    public Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      element.value = values.get( ++element.index );
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
