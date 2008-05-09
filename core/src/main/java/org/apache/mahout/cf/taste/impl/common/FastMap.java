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

import java.util.AbstractCollection;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * <p>This is an optimized {@link Map} implementation, based on algorithms described in Knuth's
 * "Art of Computer Programming", Vol. 3, p. 529.</p>
 *
 * <p>It should be faster than {@link java.util.HashMap} in some cases, but not all. Its main feature is
 * a "max size" and the ability to transparently, efficiently and semi-intelligently evict old entries
 * when max size is exceeded.</p>
 *
 * <p>This class is not a bit thread-safe.</p>
 *
 * <p>This implementation does not allow <code>null</code> as a key or value.</p>
 */
public final class FastMap<K, V> implements Map<K, V> {

  /**
   * The largest prime less than 2<sup>31</sup>-1 that is the smaller of a twin prime pair.
   */
  private static final int MAX_INT_SMALLER_TWIN_PRIME = 2147482949;

  public static final int NO_MAX_SIZE = Integer.MAX_VALUE;

  /**
   * Dummy object used to represent a key that has been removed. Package-private to allow direct access
   * by inner classes. No harm in exposing it.
   */
  private static final Object REMOVED = new Object();

  private K[] keys;
  private V[] values;
  private int numEntries;
  private int numSlotsUsed;
  private Set<Entry<K, V>> entrySet;
  private Set<K> keySet;
  private Collection<V> valueCollection;
  private int maxSize;
  private final BitSet recentlyAccessed;

  /**
   * Creates a new {@link FastMap} with default capacity.
   */
  public FastMap() {
    this(11, NO_MAX_SIZE);
  }

  public FastMap(int size) {
    this(size, NO_MAX_SIZE);
  }

  /**
   * Creates a new {@link FastMap} whose capacity can accommodate the given number of entries without rehash.</p>
   *
   * @param size desired capacity
   * @param maxSize max capacity
   * @throws IllegalArgumentException if size is less than 1 or at least half of {@link #MAX_INT_SMALLER_TWIN_PRIME}
   */
  public FastMap(int size, int maxSize) throws IllegalArgumentException {
    if (size < 1) {
      throw new IllegalArgumentException("size must be at least 1");
    }
    if (size >= MAX_INT_SMALLER_TWIN_PRIME >> 1) {
      throw new IllegalArgumentException("size must be less than " + (MAX_INT_SMALLER_TWIN_PRIME >> 1));
    }
    if (maxSize < 1) {
      throw new IllegalArgumentException("maxSize must be at least 1");
    }
    int hashSize = nextTwinPrime(2 * size);
    keys = (K[]) new Object[hashSize];
    values = (V[]) new Object[hashSize];
    this.maxSize = maxSize;
    this.recentlyAccessed = maxSize == Integer.MAX_VALUE ? null : new BitSet(maxSize);
  }

  /**
   * This is for the benefit of inner classes. Without it the compiler would just generate a similar synthetic
   * accessor. Might as well make it explicit.
   */
  K[] getKeys() {
    return keys;
  }

  /**
   * This is for the benefit of inner classes. Without it the compiler would just generate a similar synthetic
   * accessor. Might as well make it explicit.
   */
  V[] getValues() {
    return values;
  }

  private int find(Object key) {
    int theHashCode = key.hashCode() & 0x7FFFFFFF; // make sure it's positive
    K[] keys = this.keys;
    int hashSize = keys.length;
    int jump = 1 + theHashCode % (hashSize - 2);
    int index = theHashCode % hashSize;
    K currentKey = keys[index];
    while (currentKey != null && (currentKey == REMOVED || !key.equals(currentKey))) {
      if (index < jump) {
        index += hashSize - jump;
      } else {
        index -= jump;
      }
      currentKey = keys[index];
    }
    return index;
  }

  public V get(Object key) {
    if (key == null) {
      return null;
    }
    int index = find(key);
    if (recentlyAccessed != null) {
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

  public boolean containsKey(Object key) {
    return key != null && keys[find(key)] != null;
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

  /**
   * @throws NullPointerException if key or value is null
   */
  public V put(K key, V value) {
    if (key == null || value == null) {
      throw new NullPointerException();
    }
    int hashSize = keys.length;
    if (numSlotsUsed >= hashSize >> 1) {
      growAndRehash();
    }
    // Here we may later consider implementing Brent's variation described on page 532
    int index = find(key);
    if (keys[index] == null) {
      // If size is limited,
      if (recentlyAccessed != null && numEntries >= maxSize) {
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
      int hashSize = keys.length;
      K currentKey;
      do {
        if (index == 0) {
          index = hashSize - 1;
        } else {
          index--;
        }
        currentKey = keys[index];
      } while (currentKey == null || currentKey == REMOVED);
      if (recentlyAccessed.get(index)) {
        recentlyAccessed.clear(index);
      } else {
        break;
      }
    }
    // Delete the entry
    ((Object[]) keys)[index] = REMOVED;
    numEntries--;
    values[index] = null;
  }

  public void putAll(Map<? extends K, ? extends V> map) {
    if (map == null) {
      throw new NullPointerException();
    }
    for (Entry<? extends K, ? extends V> entry : map.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  public V remove(Object key) {
    if (key == null) {
      return null;
    }
    int index = find(key);
    if (keys[index] == null) {
      return null;
    } else {
      ((Object[]) keys)[index] = REMOVED;
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
    Arrays.fill(keys, null);
    Arrays.fill(values, null);
    if (recentlyAccessed != null) {
      recentlyAccessed.clear();
    }
  }

  public Set<K> keySet() {
    if (keySet == null) {
      keySet = new KeySet();
    }
    return keySet;
  }

  public Collection<V> values() {
    if (valueCollection == null) {
      valueCollection = new ValueCollection();
    }
    return valueCollection;
  }

  public Set<Entry<K, V>> entrySet() {
    if (entrySet == null) {
      entrySet = new EntrySet();
    }
    return entrySet;
  }

  public void rehash() {
    rehash(keys.length);
  }

  private void growAndRehash() {
    int hashSize = keys.length;
    if (hashSize >= MAX_INT_SMALLER_TWIN_PRIME >> 1) {
      throw new IllegalStateException("Can't grow any more");
    }
    rehash(nextTwinPrime(2 * hashSize));
  }

  private void rehash(int newHashSize) {
    K[] oldKeys = keys;
    V[] oldValues = values;
    numEntries = 0;
    numSlotsUsed = 0;
    if (recentlyAccessed != null) {
      recentlyAccessed.clear();
    }
    keys = (K[]) new Object[newHashSize];
    values = (V[]) new Object[newHashSize];
    int length = oldKeys.length;
    for (int i = 0; i < length; i++) {
      K key = oldKeys[i];
      if (key != null && key != REMOVED) {
        put(key, oldValues[i]);
      }
    }
  }

  // Simple methods for finding a next larger prime


  /**
   * <p>Finds next-largest "twin primes": numbers p and p+2 such that both are prime. Finds the smallest such p such
   * that the smaller twin, p, is greater than or equal to n. Returns p+2, the larger of the two twins.</p>
   */
  private static int nextTwinPrime(int n) {
    if (n > MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalArgumentException();
    }
    int next = nextPrime(n);
    while (isNotPrime(next + 2)) {
      next = nextPrime(next + 4);
    }
    return next + 2;
  }

  /**
   * <p>Finds smallest prime p such that p is greater than or equal to n.</p>
   */
  private static int nextPrime(int n) {
    // Make sure the number is odd. Is this too clever?
    n |= 0x1;
    // There is no problem with overflow since Integer.MAX_INT is prime, as it happens
    while (isNotPrime(n)) {
      n += 2;
    }
    return n;
  }

  /**
   * @param n
   * @return <code>true</code> iff n is not a prime
   */
  private static boolean isNotPrime(int n) {
    if (n < 2) {
      throw new IllegalArgumentException();
    }
    if ((n & 0x1) == 0) { // even
      return true;
    }
    int max = 1 + (int) Math.sqrt((double) n);
    for (int d = 3; d <= max; d += 2) {
      if (n % d == 0) {
        return true;
      }
    }
    return false;
  }

  private void iteratorRemove(int lastNext) {
    if (lastNext >= values.length) {
      throw new NoSuchElementException();
    }
    if (lastNext < 0) {
      throw new IllegalStateException();
    }
    values[lastNext] = null;
    ((Object[]) keys)[lastNext] = REMOVED;
    numEntries--;
  }

  private final class EntrySet extends AbstractSet<Entry<K, V>> {

    @Override
    public int size() {
      return FastMap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return FastMap.this.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
      return FastMap.this.containsKey(o);
    }

    @Override
    public Iterator<Entry<K, V>> iterator() {
      return new EntryIterator();
    }

    @Override
    public boolean add(Entry<K, V> t) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends Entry<K, V>> ts) {
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
      FastMap.this.clear();
    }

    final class MapEntry implements Entry<K, V> {

      private final int index;

      private MapEntry(int index) {
        this.index = index;
      }

      public K getKey() {
        return getKeys()[index];
      }

      public V getValue() {
        return getValues()[index];
      }

      public V setValue(V value) {
        if (value == null) {
          throw new IllegalArgumentException();
        }
        V[] values = getValues();
        V oldValue = values[index];
        getValues()[index] = value;
        return oldValue;
      }
    }

    final class EntryIterator implements Iterator<Entry<K, V>> {

      private int position;
      private int lastNext = -1;

      public boolean hasNext() {
        goToNext();
        return position < getKeys().length;
      }

      public Entry<K, V> next() {
        goToNext();
        lastNext = position;
        K[] keys = getKeys();
        if (position >= keys.length) {
          throw new NoSuchElementException();
        }
        return new MapEntry(position++);
      }

      private void goToNext() {
        V[] values = getValues();
        int length = values.length;
        while (position < length && values[position] == null) {
          position++;
        }
      }

      public void remove() {
        iteratorRemove(lastNext);
      }
    }

  }

  private final class KeySet extends AbstractSet<K> {

    @Override
    public int size() {
      return FastMap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return FastMap.this.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
      return FastMap.this.containsKey(o);
    }

    @Override
    public Iterator<K> iterator() {
      return new KeyIterator();
    }

    @Override
    public boolean add(K t) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends K> ts) {
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
      FastMap.this.clear();
    }

    final class KeyIterator implements Iterator<K> {

      private int position;
      private int lastNext = -1;

      public boolean hasNext() {
        goToNext();
        return position < getKeys().length;
      }

      public K next() {
        goToNext();
        lastNext = position;
        K[] keys = getKeys();
        if (position >= keys.length) {
          throw new NoSuchElementException();
        }
        return keys[position++];
      }

      private void goToNext() {
        V[] values = getValues();
        int length = values.length;
        while (position < length && values[position] == null) {
          position++;
        }
      }

      public void remove() {
        iteratorRemove(lastNext);
      }
    }

  }

  private final class ValueCollection extends AbstractCollection<V> {

    @Override
    public int size() {
      return FastMap.this.size();
    }

    @Override
    public boolean isEmpty() {
      return FastMap.this.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
      return FastMap.this.containsValue(o);
    }

    @Override
    public Iterator<V> iterator() {
      return new ValueIterator();
    }

    @Override
    public boolean add(V v) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends V> vs) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> objects) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> objects) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      FastMap.this.clear();
    }

    final class ValueIterator implements Iterator<V> {

      private int position;
      private int lastNext = -1;

      public boolean hasNext() {
        goToNext();
        return position < getValues().length;
      }

      public V next() {
        goToNext();
        lastNext = position;
        V[] values = getValues();
        if (position >= values.length) {
          throw new NoSuchElementException();
        }
        return values[position++];
      }

      private void goToNext() {
        V[] values = getValues();
        int length = values.length;
        while (position < length && values[position] == null) {
          position++;
        }
      }

      public void remove() {
        iteratorRemove(lastNext);
      }

    }

  }
}
