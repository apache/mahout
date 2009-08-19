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
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * @see FastByIDMap
 */
public final class FastIDSet implements Serializable, Cloneable {

  private static final double ALLOWED_LOAD_FACTOR = 1.5;

  /** Dummy object used to represent a key that has been removed. */
  private static final long REMOVED = Long.MAX_VALUE;
  private static final long NULL = Long.MIN_VALUE;

  private long[] keys;
  private int numEntries;
  private int numSlotsUsed;

  /** Creates a new {@link FastIDSet} with default capacity. */
  public FastIDSet() {
    this(2);
  }

  public FastIDSet(int size) {
    if (size < 0) {
      throw new IllegalArgumentException("size must be at least 0");
    }
    int max = (int) (RandomUtils.MAX_INT_SMALLER_TWIN_PRIME / ALLOWED_LOAD_FACTOR);
    if (size >= max) {
      throw new IllegalArgumentException("size must be less than " + max);
    }
    int hashSize = RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * size));
    keys = new long[hashSize];
    Arrays.fill(keys, NULL);
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

  public int size() {
    return numEntries;
  }

  public boolean isEmpty() {
    return numEntries == 0;
  }

  public boolean contains(long key) {
    return key != NULL && key != REMOVED && keys[find(key)] != NULL;
  }

  public boolean add(long key) {
    if (key == NULL || key == REMOVED) {
      throw new IllegalArgumentException();
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
      keys[index] = key;
      numEntries++;
      numSlotsUsed++;
      return true;
    }
    return false;
  }

  public LongPrimitiveIterator iterator() {
    return new KeyIterator();
  }

  public long[] toArray() {
    long[] result = new long[numEntries];
    for (int i = 0, position = 0; i < result.length; i++) {
      while (keys[position] == NULL || keys[position] == REMOVED) {
        position++;
      }
      result[i] = keys[position++];
    }
    return result;
  }

  public boolean remove(long key) {
    if (key == NULL || key == REMOVED) {
      return false;
    }
    int index = find(key);
    if (keys[index] == NULL) {
      return false;
    } else {
      keys[index] = REMOVED;
      numEntries--;
      return true;
    }
  }

  public boolean addAll(long[] c) {
    boolean changed = false;
    for (long k : c) {
      if (add(k)) {
        changed = true;
      }
    }
    return changed;
  }

  public boolean addAll(FastIDSet c) {
    boolean changed = false;
    for (long k : c.keys) {
      if (k != NULL && k != REMOVED) {
        if (add(k)) {
          changed = true;
        }
      }
    }
    return changed;
  }

  public boolean removeAll(long[] c) {
    boolean changed = false;
    for (long o : c) {
      if (remove(o)) {
        changed = true;
      }
    }
    return changed;
  }

  public boolean removeAll(FastIDSet c) {
    boolean changed = false;
    for (long k : c.keys) {
      if (k != NULL && k != REMOVED) {
        if (remove(k)) {
          changed = true;
        }
      }
    }
    return changed;
  }

  public boolean retainAll(FastIDSet c) {
    boolean changed = false;
    for (int i = 0; i < keys.length; i++) {
      long k = keys[i];
      if (k != NULL && k != REMOVED && !c.contains(k)) {
        keys[i] = REMOVED;
        numEntries--;
        changed = true;
      }
    }
    return changed;
  }

  public void clear() {
    numEntries = 0;
    numSlotsUsed = 0;
    Arrays.fill(keys, NULL);
  }

  private void growAndRehash() {
    if (keys.length * ALLOWED_LOAD_FACTOR >= RandomUtils.MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalStateException("Can't grow any more");
    }
    rehash(RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * keys.length)));
  }

  public void rehash() {
    rehash(RandomUtils.nextTwinPrime((int) (ALLOWED_LOAD_FACTOR * numEntries)));
  }

  private void rehash(int newHashSize) {
    long[] oldKeys = keys;
    numEntries = 0;
    numSlotsUsed = 0;
    keys = new long[newHashSize];
    Arrays.fill(keys, NULL);
    int length = oldKeys.length;
    for (int i = 0; i < length; i++) {
      long key = oldKeys[i];
      if (key != NULL && key != REMOVED) {
        add(key);
      }
    }
  }

  /**
   * Convenience method to quickly compute just the size of the intersection with another {@link FastIDSet}.
   *
   * @param other {@link FastIDSet} to intersect with
   * @return number of elements in intersection
   */
  public int intersectionSize(FastIDSet other) {
    int count = 0;
    for (long key : other.keys) {
      if (key != NULL && key != REMOVED && keys[find(key)] != NULL) {
        count++;
      }
    }
    return count;
  }

  @Override
  public FastIDSet clone() {
    FastIDSet clone;
    try {
      clone = (FastIDSet) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new AssertionError();
    }
    clone.keys = keys.clone();
    return clone;
  }

  @Override
  public String toString() {
    if (isEmpty()) {
      return "[]";
    }
    StringBuilder result = new StringBuilder();
    result.append('[');
    for (long key : keys) {
      if (key != NULL && key != REMOVED) {
        result.append(key).append(',');
      }
    }
    result.setCharAt(result.length() - 1, ']');
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
      int length = keys.length;
      while (position < length && (keys[position] == NULL || keys[position] == REMOVED)) {
        position++;
      }
    }

    @Override
    public void remove() {
      if (lastNext >= keys.length) {
        throw new NoSuchElementException();
      }
      if (lastNext < 0) {
        throw new IllegalStateException();
      }
      keys[lastNext] = REMOVED;
      numEntries--;
    }

    public Iterator<Long> iterator() {
      return new KeyIterator();
    }

  }

}