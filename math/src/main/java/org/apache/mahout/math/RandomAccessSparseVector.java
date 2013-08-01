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

import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.map.OpenIntDoubleHashMap.MapElement;
import org.apache.mahout.math.set.AbstractSet;


/** Implements vector that only stores non-zero doubles */
public class RandomAccessSparseVector extends AbstractVector {

  private static final int INITIAL_CAPACITY = 11;

  private OpenIntDoubleHashMap values;

  /** For serialization purposes only. */
  public RandomAccessSparseVector() {
    super(0);
  }

  public RandomAccessSparseVector(int cardinality) {
    this(cardinality, Math.min(cardinality, INITIAL_CAPACITY)); // arbitrary estimate of 'sparseness'
  }

  public RandomAccessSparseVector(int cardinality, int initialCapacity) {
    super(cardinality);
    values = new OpenIntDoubleHashMap(initialCapacity);
  }

  public RandomAccessSparseVector(Vector other) {
    this(other.size(), other.getNumNondefaultElements());
    for (Element e : other.nonZeroes()) {
      values.put(e.index(), e.get());
    }
  }

  private RandomAccessSparseVector(int cardinality, OpenIntDoubleHashMap values) {
    super(cardinality);
    this.values = values;
  }

  public RandomAccessSparseVector(RandomAccessSparseVector other, boolean shallowCopy) {
    super(other.size());
    values = shallowCopy ? other.values : (OpenIntDoubleHashMap)other.values.clone();
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new SparseRowMatrix(rows, columns);
  }

  @Override
  public RandomAccessSparseVector clone() {
    return new RandomAccessSparseVector(size(), (OpenIntDoubleHashMap) values.clone());
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
      values.removeKey(index);
    } else {
      values.put(index, value);
    }
  }

  @Override
  public void incrementQuick(int index, double increment) {
    invalidateCachedLength();
    values.adjustOrPutValue(index, increment, increment);
  }


  @Override
  public RandomAccessSparseVector like() {
    return new RandomAccessSparseVector(size(), values.size());
  }

  @Override
  public int getNumNondefaultElements() {
    return values.size();
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

  /**
   * NOTE: this implementation reuses the Vector.Element instance for each call of next(). If you need to preserve the
   * instance, you need to make a copy of it
   *
   * @return an {@link Iterator} over the Elements.
   * @see #getElement(int)
   */
  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonDefaultIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  private final class NonDefaultIterator implements Iterator<Element> {
    private final class NonDefaultElement implements Element {
      @Override
      public double get() {
        return mapElement.get();
      }

      @Override
      public int index() {
        return mapElement.index();
      }

      @Override
      public void set(double value) {
        invalidateCachedLength();
        mapElement.set(value);
      }
    }


    private MapElement mapElement;
    private final NonDefaultElement element = new NonDefaultElement();

    private final Iterator<MapElement> iterator;

    private NonDefaultIterator() {
      this.iterator = values.iterator();
    }

    @Override
    public boolean hasNext() {
      return iterator.hasNext();
    }

    @Override
    public Element next() {
      mapElement = iterator.next(); // This will throw an exception at the end of enumeration.
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class AllIterator implements Iterator<Element> {
    private final RandomAccessElement element = new RandomAccessElement();

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
      element.index++;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class RandomAccessElement implements Element {
    int index;

    @Override
    public double get() {
      return values.get(index);
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      invalidateCachedLength();
      if (value == 0.0) {
        values.removeKey(index);
      } else {
        values.put(index, value);
      }
    }
  }
}
