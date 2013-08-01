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

import com.google.common.primitives.Doubles;
import org.apache.mahout.math.function.Functions;

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
 *
 * See {@link OrderedIntDoubleMapping}
 */
public class SequentialAccessSparseVector extends AbstractVector {

  private OrderedIntDoubleMapping values;

  /** For serialization purposes only. */
  public SequentialAccessSparseVector() {
    super(0);
  }

  public SequentialAccessSparseVector(int cardinality) {
    this(cardinality, Math.min(100, cardinality / 1000 < 10 ? 10 : cardinality / 1000)); // arbitrary estimate of
                                                                                           // 'sparseness'
  }

  public SequentialAccessSparseVector(int cardinality, int size) {
    super(cardinality);
    values = new OrderedIntDoubleMapping(size);
  }

  public SequentialAccessSparseVector(Vector other) {
    this(other.size(), other.getNumNondefaultElements());

    if (other.isSequentialAccess()) {
      for (Element e : other.nonZeroes()) {
        set(e.index(), e.get());
      }
    } else {
      // If the incoming Vector to copy is random, then adding items
      // from the Iterator can degrade performance dramatically if
      // the number of elements is large as this Vector tries to stay
      // in order as items are added, so it's better to sort the other
      // Vector's elements by index and then add them to this
      copySortedRandomAccessSparseVector(other);
    }
  }

  // Sorts a RandomAccessSparseVectors Elements before adding them to this
  private int copySortedRandomAccessSparseVector(Vector other) {
    int elementCount = other.getNumNondefaultElements();
    OrderedElement[] sortableElements = new OrderedElement[elementCount];
    int s = 0;
    for (Element e : other.nonZeroes()) {
      sortableElements[s++] = new OrderedElement(e.index(), e.get());
    }
    Arrays.sort(sortableElements);
    for (int i = 0; i < sortableElements.length; i++) {
      values.setIndexAt(i, sortableElements[i].index);
      values.setValueAt(i, sortableElements[i].value);
    }
    values = new OrderedIntDoubleMapping(values.getIndices(), values.getValues(), elementCount);
    return elementCount;
  }

  public SequentialAccessSparseVector(SequentialAccessSparseVector other, boolean shallowCopy) {
    super(other.size());
    values = shallowCopy ? other.values : other.values.clone();
  }

  public SequentialAccessSparseVector(SequentialAccessSparseVector other) {
    this(other.size(), other.getNumNondefaultElements());
    values = other.values.clone();
  }

  private SequentialAccessSparseVector(int cardinality, OrderedIntDoubleMapping values) {
    super(cardinality);
    this.values = values;
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new SparseRowMatrix(rows, columns);
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public SequentialAccessSparseVector clone() {
    return new SequentialAccessSparseVector(size(), values.clone());
  }

  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    values.merge(updates);
  }

  @Override
  public String toString() {
    return sparseVectorToString();
  }

  /**
   * @return false
   */
  @Override
  public boolean isDense() {
    return false;
  }

  /**
   * @return true
   */
  @Override
  public boolean isSequentialAccess() {
    return true;
  }

  /**
   * Warning! This takes O(log n) time as it does a binary search behind the scenes!
   * Only use it when STRICTLY necessary.
   * @param index an int index.
   * @return the value at that position in the vector.
   */
  @Override
  public double getQuick(int index) {
    return values.get(index);
  }

  /**
   * Warning! This takes O(log n) time as it does a binary search behind the scenes!
   * Only use it when STRICTLY necessary.
   * @param index an int index.
   */
  @Override
  public void setQuick(int index, double value) {
    invalidateCachedLength();
    values.set(index, value);
  }

  @Override
  public void incrementQuick(int index, double increment) {
    invalidateCachedLength();
    values.increment(index, increment);
  }

  @Override
  public SequentialAccessSparseVector like() {
    return new SequentialAccessSparseVector(size(), values.getNumMappings());
  }

  @Override
  public int getNumNondefaultElements() {
    return values.getNumMappings();
  }

  @Override
  public double getLookupCost() {
    return Math.max(1, Math.round(Functions.LOG2.apply(getNumNondefaultElements())));
  }

  @Override
  public double getIteratorAdvanceCost() {
    return 1;
  }

  @Override
  public boolean isAddConstantTime() {
    return false;
  }

  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonDefaultIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  private final class NonDefaultIterator implements Iterator<Element> {
    private final NonDefaultElement element = new NonDefaultElement();

    @Override
    public boolean hasNext() {
      return element.getNextOffset() < values.getNumMappings();
    }

    @Override
    public Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      element.advanceOffset();
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class AllIterator implements Iterator<Element> {
    private final AllElement element = new AllElement();

    @Override
    public boolean hasNext() {
      return element.getNextIndex() < SequentialAccessSparseVector.this.size();
    }

    @Override
    public Element next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }

      element.advanceIndex();
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class NonDefaultElement implements Element {
    private int offset = -1;

    void advanceOffset() {
      offset++;
    }

    int getNextOffset() {
      return offset + 1;
    }

    @Override
    public double get() {
      return values.getValues()[offset];
    }

    @Override
    public int index() {
      return values.getIndices()[offset];
    }

    @Override
    public void set(double value) {
      invalidateCachedLength();
      values.setValueAt(offset, value);
    }
  }

  private final class AllElement implements Element {
    private int index = -1;
    private int nextOffset;

    void advanceIndex() {
      index++;
      if (nextOffset < values.getNumMappings() && index > values.getIndices()[nextOffset]) {
        nextOffset++;
      }
    }

    int getNextIndex() {
      return index + 1;
    }

    @Override
    public double get() {
      if (nextOffset < values.getNumMappings() && index == values.getIndices()[nextOffset]) {
        return values.getValues()[nextOffset];
      } else {
        return OrderedIntDoubleMapping.DEFAULT_VALUE;
      }
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      invalidateCachedLength();
      if (nextOffset < values.getNumMappings() && index == values.indexAt(nextOffset)) {
        values.setValueAt(nextOffset, value);
      } else {
        // Yes, this works; the offset into indices of the new value's index will still be nextOffset
        values.set(index, value);
      }
    }
  }

  // Comparable Element for sorting Elements by index
  private static final class OrderedElement implements Comparable<OrderedElement> {
    private final int index;
    private final double value;

    OrderedElement(int index, double value) {
      this.index = index;
      this.value = value;
    }

    @Override
    public int compareTo(OrderedElement that) {
      // both indexes are positive, and neither can be Integer.MAX_VALUE (otherwise there would be
      // an array somewhere with Integer.MAX_VALUE + 1 elements)
      return this.index - that.index;
    }

    @Override
    public int hashCode() {
      return index ^ Doubles.hashCode(value);
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof OrderedElement)) {
        return false;
      }
      OrderedElement other = (OrderedElement) o;
      return index == other.index && value == other.value;
    }
  }
}
