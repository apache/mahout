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

package org.apache.mahout.cf.taste.impl.common;

import java.io.Serializable;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * @see FastMap
 * @see FastIDSet
 */
public final class FastByIDMap<V> implements Serializable, Cloneable {

  public static final int NO_MAX_SIZE = Integer.MAX_VALUE;
  private static final double ALLOWED_LOAD_FACTOR = 1.5;

  /** Dummy object used to represent a key that has been removed. */
  private static final long REMOVED = Long.MAX_VALUE;
  private static final long NULL = Long.MIN_VALUE;

  private long[] keys;
  private V[] values;
  private int numEntries;
  private int numSlotsUsed;
  private int maxSize;
  private BitSet recentlyAccessed;
  private final boolean countingAccesses;

  /** Creates a new {@link FastByIDMap} with default capacity. */
  public FastByIDMap() {
    this(2, NO_MAX_SIZE);
  }

  public FastByIDMap(int size) {
    this(size, NO_MAX_SIZE);
  }

  /**
   * Creates a new {@link FastByIDMap} whose capacity can accommodate the given number of entries without rehash.</p>
   *
   * @param size    desired capacity
   * @param maxSize max capacity
   * @throws IllegalArgumentException if size is less than 0, maxSize is less than 1,
   *  or at least half of {@link RandomUtils#MAX_INT_SMALLER_TWIN_PRIME}
   */
  @SuppressWarnings("unchecked")
  public FastByIDMap(int size, int maxSize) {
    if (size < 0) {
      throw new IllegalArgumentException("size must be at least 0");
    }
    int max = (int) (RandomUtils.MAX_INT_SMALLER_TWIN_PRIME / ALLOWED_LOAD_FACTOR);
    if (size >= max) {
      throw new IllegalArgumentException("size must be less than " + max);
    }
    if (maxSize < 1) {
      throw new IllegalArgumentException("maxSize must be at least 1");
    }
    int hashSize = RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * size));
    keys = new long[hashSize];
    Arrays.fill(keys, NULL);
    values = (V[]) new Object[hashSize];
    this.maxSize = maxSize;
    this.countingAccesses = maxSize != Integer.MAX_VALUE;
    this.recentlyAccessed = countingAccesses ? new BitSet(hashSize) : null;
  }

  private int find(long key) {
    int theHashCode = (int) key & 0x7FFFFFFF; // make sure it's positive
    long[] keys = this.keys;
    int hashSize = keys.length;
    int jump = 1 + theHashCode % (hashSize - 2);
    int index = theHashCode % hashSize;
    long currentKey = keys[index];
    while (currentKey != NULL && (currentKey == REMOVED || key != currentKey)) {
      if (index < jump) {
        index += hashSize - jump;
      } else {
        index -= jump;
      }
      currentKey = keys[index];
    }
    return index;
  }

  public V get(long key) {
    if (key == NULL) {
      return null;
    }
    int index = find(key);
    if (countingAccesses) {
      recentlyAccessed.set(index);
    }
    return values[index];
  }

  public int size() {
    return numEntries;
  }

  public boolean isEmpty() {
    return numEntries == 0;
  }

  public boolean containsKey(long key) {
    return key != NULL && key != REMOVED && keys[find(key)] != NULL;
  }

  public boolean containsValue(Object value) {
    if (value == null) {
      return false;
    }
    for (V theValue : values) {
      if (theValue != null && value.equals(theValue)) {
        return true;
      }
    }
    return false;
  }

  public V put(long key, V value) {
    if (key == NULL || key == REMOVED) {
      throw new IllegalArgumentException();
    }
    if (value == null) {
      throw new NullPointerException();
    }
    // If less than half the slots are open, let's clear it up
    if (numSlotsUsed * ALLOWED_LOAD_FACTOR >= keys.length) {
      // If over half the slots used are actual entries, let's grow
      if (numEntries * ALLOWED_LOAD_FACTOR >= numSlotsUsed) {
        growAndRehash();
      } else {
        // Otherwise just rehash to clear REMOVED entries and don't grow
        rehash();
      }
    }
    // Here we may later consider implementing Brent's variation described on page 532
    int index = find(key);
    if (keys[index] == NULL) {
      // If size is limited,
      if (countingAccesses && numEntries >= maxSize) {
        // and we're too large, clear some old-ish entry
        clearStaleEntry(index);
      }
      keys[index] = key;
      values[index] = value;
      numEntries++;
      numSlotsUsed++;
      return null;
    } else {
      V oldValue = values[index];
      values[index] = value;
      return oldValue;
    }
  }

  private void clearStaleEntry(int index) {
    while (true) {
      long currentKey;
      do {
        if (index == 0) {
          index = keys.length - 1;
        } else {
          index--;
        }
        currentKey = keys[index];
      } while (currentKey == NULL || currentKey == REMOVED);
      if (recentlyAccessed.get(index)) {
        recentlyAccessed.clear(index);
      } else {
        break;
      }
    }
    // Delete the entry
    keys[index] = REMOVED;
    numEntries--;
    values[index] = null;
  }

  public V remove(long key) {
    if (key == NULL || key == REMOVED) {
      return null;
    }
    int index = find(key);
    if (keys[index] == NULL) {
      return null;
    } else {
      keys[index] = REMOVED;
      numEntries--;
      V oldValue = values[index];
      values[index] = null;
      // don't decrement numSlotsUsed
      return oldValue;
    }
    // Could un-set recentlyAccessed's bit but doesn't matter
  }

  public void clear() {
    numEntries = 0;
    numSlotsUsed = 0;
    Arrays.fill(keys, NULL);
    Arrays.fill(values, null);
    if (countingAccesses) {
      recentlyAccessed.clear();
    }
  }

  public LongPrimitiveIterator keySetIterator() {
    return new KeyIterator();
  }

  public Set<Map.Entry<Long, V>> entrySet() {
    return new EntrySet();
  }

  public void rehash() {
    rehash(RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * numEntries)));
  }

  private void growAndRehash() {
    if (keys.length * ALLOWED_LOAD_FACTOR >= RandomUtils.MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalStateException("Can't grow any more");
    }
    rehash(RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * keys.length)));
  }

  private void rehash(int newHashSize) {
    long[] oldKeys = keys;
    V[] oldValues = values;
    numEntries = 0;
    numSlotsUsed = 0;
    if (countingAccesses) {
      recentlyAccessed = new BitSet(newHashSize);
    }
    keys = new long[newHashSize];
    Arrays.fill(keys, NULL);
    values = (V[]) new Object[newHashSize];
    int length = oldKeys.length;
    for (int i = 0; i < length; i++) {
      long key = oldKeys[i];
      if (key != NULL && key != REMOVED) {
        put(key, oldValues[i]);
      }
    }
  }

  void iteratorRemove(int lastNext) {
    if (lastNext >= values.length) {
      throw new NoSuchElementException();
    }
    if (lastNext < 0) {
      throw new IllegalStateException();
    }
    values[lastNext] = null;
    keys[lastNext] = REMOVED;
    numEntries--;
  }

  @Override
  public FastByIDMap<V> clone() {
    FastByIDMap<V> clone;
    try {
      clone = (FastByIDMap<V>) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new AssertionError();
    }
    clone.keys = keys.clone();
    clone.values = values.clone();
    clone.recentlyAccessed = countingAccesses ? new BitSet(keys.length) : null;
    return clone;
  }

  @Override
  public String toString() {
    if (isEmpty()) {
      return "{}";
    }
    StringBuilder result = new StringBuilder();
    result.append('{');
    for (int i = 0; i < keys.length; i++) {
      long key = keys[i];
      if (key != NULL && key != REMOVED) {
        result.append(key).append('=').append(values[i]).append(',');
      }
    }
    result.setCharAt(result.length() - 1, '}');
    return result.toString();
  }

  private final class KeyIterator implements LongPrimitiveIterator {

    private int position;
    private int lastNext = -1;

    @Override
    public boolean hasNext() {
      goToNext();
      return position < keys.length;
    }

    @Override
    public Long next() {
      return nextLong();
    }

    @Override
    public long nextLong() {
      goToNext();
      lastNext = position;
      if (position >= keys.length) {
        throw new NoSuchElementException();
      }
      return keys[position++];
    }

    @Override
    public long peek() {
      goToNext();
      if (position >= keys.length) {
        throw new NoSuchElementException();
      }
      return keys[position];
    }

    private void goToNext() {
      int length = values.length;
      while (position < length && values[position] == null) {
        position++;
      }
    }

    @Override
    public void remove() {
      iteratorRemove(lastNext);
    }
  }

  private final class EntrySet extends AbstractSet<Map.Entry<Long, V>> {

    @Override
    public int size() {
      return FastByIDMap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return FastByIDMap.this.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
      return containsKey((Long) o);
    }

    @Override
    public Iterator<Map.Entry<Long, V>> iterator() {
      return new EntryIterator();
    }

    @Override
    public boolean add(Map.Entry<Long, V> t) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends Map.Entry<Long, V>> ts) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> objects) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> objects) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      FastByIDMap.this.clear();
    }

    private final class MapEntry implements Map.Entry<Long, V> {

      private final int index;

      private MapEntry(int index) {
        this.index = index;
      }

      @Override
      public Long getKey() {
        return keys[index];
      }

      @Override
      public V getValue() {
        return values[index];
      }

      @Override
      public V setValue(V value) {
        if (value == null) {
          throw new IllegalArgumentException();
        }
        V oldValue = values[index];
        values[index] = value;
        return oldValue;
      }
    }

    private final class EntryIterator implements Iterator<Map.Entry<Long, V>> {

      private int position;
      private int lastNext = -1;

      @Override
      public boolean hasNext() {
        goToNext();
        return position < keys.length;
      }

      @Override
      public Map.Entry<Long, V> next() {
        goToNext();
        lastNext = position;
        if (position >= keys.length) {
          throw new NoSuchElementException();
        }
        return new MapEntry(position++);
      }

      private void goToNext() {
        int length = values.length;
        while (position < length && values[position] == null) {
          position++;
        }
      }

      @Override
      public void remove() {
        iteratorRemove(lastNext);
      }
    }

  }

}