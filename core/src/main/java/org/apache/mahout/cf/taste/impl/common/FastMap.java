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
import java.util.AbstractCollection;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import org.apache.mahout.common.RandomUtils;

import com.google.common.base.Preconditions;

/**
 * <p>
 * This is an optimized {@link Map} implementation, based on algorithms described in Knuth's "Art of Computer
 * Programming", Vol. 3, p. 529.
 * </p>
 *
 * <p>
 * It should be faster than {@link java.util.HashMap} in some cases, but not all. Its main feature is a
 * "max size" and the ability to transparently, efficiently and semi-intelligently evict old entries when max
 * size is exceeded.
 * </p>
 *
 * <p>
 * This class is not a bit thread-safe.
 * </p>
 *
 * <p>
 * This implementation does not allow {@code null} as a key or value.
 * </p>
 */
public final class FastMap<K,V> implements Map<K,V>, Serializable, Cloneable {
  
  public static final int NO_MAX_SIZE = Integer.MAX_VALUE;
  private static final float DEFAULT_LOAD_FACTOR = 1.5f;
  
  /** Dummy object used to represent a key that has been removed. */
  private static final Object REMOVED = new Object();
  
  private K[] keys;
  private V[] values;
  private float loadFactor;
  private int numEntries;
  private int numSlotsUsed;
  private final int maxSize;
  private BitSet recentlyAccessed;
  private final boolean countingAccesses;
  
  /** Creates a new {@link FastMap} with default capacity. */
  public FastMap() {
    this(2, NO_MAX_SIZE);
  }
  
  public FastMap(int size) {
    this(size, NO_MAX_SIZE);
  }
  
  public FastMap(Map<K,V> other) {
    this(other.size());
    putAll(other);
  }

  public FastMap(int size, float loadFactor) {
    this(size, NO_MAX_SIZE, loadFactor);
  }

  public FastMap(int size, int maxSize) {
    this(size, maxSize, DEFAULT_LOAD_FACTOR);
  }
  
  /**
   * Creates a new  whose capacity can accommodate the given number of entries without rehash.
   * 
   * @param size desired capacity
   * @param maxSize max capacity
   * @throws IllegalArgumentException if size is less than 0, maxSize is less than 1
   *  or at least half of {@link RandomUtils#MAX_INT_SMALLER_TWIN_PRIME}, or
   *  loadFactor is less than 1
   */
  public FastMap(int size, int maxSize, float loadFactor) {
    Preconditions.checkArgument(size >= 0, "size must be at least 0");
    Preconditions.checkArgument(loadFactor >= 1.0f, "loadFactor must be at least 1.0");
    this.loadFactor = loadFactor;
    int max = (int) (RandomUtils.MAX_INT_SMALLER_TWIN_PRIME / loadFactor);
    Preconditions.checkArgument(size < max, "size must be less than " + max);
    Preconditions.checkArgument(maxSize >= 1, "maxSize must be at least 1");
    int hashSize = RandomUtils.nextTwinPrime((int) (loadFactor * size));
    keys = (K[]) new Object[hashSize];
    values = (V[]) new Object[hashSize];
    this.maxSize = maxSize;
    this.countingAccesses = maxSize != Integer.MAX_VALUE;
    this.recentlyAccessed = countingAccesses ? new BitSet(hashSize) : null;
  }
  
  private int find(Object key) {
    int theHashCode = key.hashCode() & 0x7FFFFFFF; // make sure it's positive
    K[] keys = this.keys;
    int hashSize = keys.length;
    int jump = 1 + theHashCode % (hashSize - 2);
    int index = theHashCode % hashSize;
    K currentKey = keys[index];
    while (currentKey != null && !key.equals(currentKey)) {
      index -= index < jump ? jump - hashSize : jump;
      currentKey = keys[index];
    }
    return index;
  }

  private int findForAdd(Object key) {
    int theHashCode = key.hashCode() & 0x7FFFFFFF; // make sure it's positive
    K[] keys = this.keys;
    int hashSize = keys.length;
    int jump = 1 + theHashCode % (hashSize - 2);
    int index = theHashCode % hashSize;
    K currentKey = keys[index];
    while (currentKey != null && currentKey != REMOVED && key != currentKey) {
      index -= index < jump ? jump - hashSize : jump;
      currentKey = keys[index];
    }
    if (currentKey != REMOVED) {
      return index;
    }
    // If we're adding, it's here, but, the key might have a value already later
    int addIndex = index;
    while (currentKey != null && key != currentKey) {
      index -= index < jump ? jump - hashSize : jump;
      currentKey = keys[index];
    }
    return key == currentKey ? index : addIndex;
  }
  
  @Override
  public V get(Object key) {
    if (key == null) {
      return null;
    }
    int index = find(key);
    if (countingAccesses) {
      recentlyAccessed.set(index);
    }
    return values[index];
  }
  
  @Override
  public int size() {
    return numEntries;
  }
  
  @Override
  public boolean isEmpty() {
    return numEntries == 0;
  }
  
  @Override
  public boolean containsKey(Object key) {
    return key != null && keys[find(key)] != null;
  }
  
  @Override
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
   * @throws NullPointerException
   *           if key or value is null
   */
  @Override
  public V put(K key, V value) {
    Preconditions.checkNotNull(key);
    Preconditions.checkNotNull(value);
    // If less than half the slots are open, let's clear it up
    if (numSlotsUsed * loadFactor >= keys.length) {
      // If over half the slots used are actual entries, let's grow
      if (numEntries * loadFactor >= numSlotsUsed) {
        growAndRehash();
      } else {
        // Otherwise just rehash to clear REMOVED entries and don't grow
        rehash();
      }
    }
    // Here we may later consider implementing Brent's variation described on page 532
    int index = findForAdd(key);
    if (keys[index] == key) {
      V oldValue = values[index];
      values[index] = value;
      return oldValue;
    }
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
  }
  
  private void clearStaleEntry(int index) {
    while (true) {
      K currentKey;
      do {
        if (index == 0) {
          index = keys.length - 1;
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
    ((Object[])keys)[index] = REMOVED;
    numEntries--;
    values[index] = null;
  }
  
  @Override
  public void putAll(Map<? extends K,? extends V> map) {
    for (Entry<? extends K,? extends V> entry : map.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }
  
  @Override
  public V remove(Object key) {
    if (key == null) {
      return null;
    }
    int index = find(key);
    if (keys[index] == null) {
      return null;
    } else {
      ((Object[])keys)[index] = REMOVED;
      numEntries--;
      V oldValue = values[index];
      values[index] = null;
      // don't decrement numSlotsUsed
      return oldValue;
    }
    // Could un-set recentlyAccessed's bit but doesn't matter
  }
  
  @Override
  public void clear() {
    numEntries = 0;
    numSlotsUsed = 0;
    Arrays.fill(keys, null);
    Arrays.fill(values, null);
    if (countingAccesses) {
      recentlyAccessed.clear();
    }
  }
  
  @Override
  public Set<K> keySet() {
    return new KeySet();
  }
  
  @Override
  public Collection<V> values() {
    return new ValueCollection();
  }
  
  @Override
  public Set<Entry<K,V>> entrySet() {
    return new EntrySet();
  }
  
  public void rehash() {
    rehash(RandomUtils.nextTwinPrime((int) (loadFactor * numEntries)));
  }
  
  private void growAndRehash() {
    if (keys.length * loadFactor >= RandomUtils.MAX_INT_SMALLER_TWIN_PRIME) {
      throw new IllegalStateException("Can't grow any more");
    }
    rehash(RandomUtils.nextTwinPrime((int) (loadFactor * keys.length)));
  }
  
  private void rehash(int newHashSize) {
    K[] oldKeys = keys;
    V[] oldValues = values;
    numEntries = 0;
    numSlotsUsed = 0;
    if (countingAccesses) {
      recentlyAccessed = new BitSet(newHashSize);
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
  
  void iteratorRemove(int lastNext) {
    if (lastNext >= values.length) {
      throw new NoSuchElementException();
    }
    if (lastNext < 0) {
      throw new IllegalStateException();
    }
    values[lastNext] = null;
    ((Object[])keys)[lastNext] = REMOVED;
    numEntries--;
  }
  
  @Override
  public FastMap<K,V> clone() {
    FastMap<K,V> clone;
    try {
      clone = (FastMap<K,V>) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new AssertionError();
    }
    clone.keys = keys.clone();
    clone.values = values.clone();
    clone.recentlyAccessed = countingAccesses ? new BitSet(keys.length) : null;
    return clone;
  }

  @Override
  public int hashCode() {
    int hash = 0;
    K[] keys = this.keys;
    int max = keys.length;
    for (int i = 0; i < max; i++) {
      K key = keys[i];
      if (key != null && key != REMOVED) {
        hash = 31 * hash + key.hashCode();
        hash = 31 * hash + values[i].hashCode();
      }
    }
    return hash;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof FastMap)) {
      return false;
    }
    FastMap<K,V> otherMap = (FastMap<K,V>) other;
    K[] otherKeys = otherMap.keys;
    V[] otherValues = otherMap.values;
    int length = keys.length;
    int otherLength = otherKeys.length;
    int max = Math.min(length, otherLength);

    int i = 0;
    while (i < max) {
      K key = keys[i];
      K otherKey = otherKeys[i];
      if (key == null || key == REMOVED) {
        if (otherKey != null && otherKey != REMOVED) {
          return false;
        }
      } else {
        if (key != otherKey || !values[i].equals(otherValues[i])) {
          return false;
        }
      }
      i++;
    }
    while (i < length) {
      K key = keys[i];
      if (key != null && key != REMOVED) {
        return false;
      }
      i++;
    }
    while (i < otherLength) {
      K key = otherKeys[i];
      if (key != null && key != REMOVED) {
        return false;
      }
      i++;
    }
    return true;
  }
  
  @Override
  public String toString() {
    if (isEmpty()) {
      return "{}";
    }
    StringBuilder result = new StringBuilder();
    result.append('{');
    for (int i = 0; i < keys.length; i++) {
      K key = keys[i];
      if (key != null && key != REMOVED) {
        result.append(key).append('=').append(values[i]).append(',');
      }
    }
    result.setCharAt(result.length() - 1, '}');
    return result.toString();
  }
  
  private final class EntrySet extends AbstractSet<Entry<K,V>> {
    
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
      return containsKey(o);
    }
    
    @Override
    public Iterator<Entry<K,V>> iterator() {
      return new EntryIterator();
    }
    
    @Override
    public boolean add(Entry<K,V> t) {
      throw new UnsupportedOperationException();
    }
    
    @Override
    public boolean remove(Object o) {
      throw new UnsupportedOperationException();
    }
    
    @Override
    public boolean addAll(Collection<? extends Entry<K,V>> ts) {
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
    
    private final class MapEntry implements Entry<K,V> {
      
      private final int index;
      
      private MapEntry(int index) {
        this.index = index;
      }
      
      @Override
      public K getKey() {
        return keys[index];
      }
      
      @Override
      public V getValue() {
        return values[index];
      }
      
      @Override
      public V setValue(V value) {
        Preconditions.checkArgument(value != null);
        V oldValue = values[index];
        values[index] = value;
        return oldValue;
      }
    }
    
    private final class EntryIterator implements Iterator<Entry<K,V>> {
      
      private int position;
      private int lastNext = -1;
      
      @Override
      public boolean hasNext() {
        goToNext();
        return position < keys.length;
      }
      
      @Override
      public Entry<K,V> next() {
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
      return containsKey(o);
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
    
    private final class KeyIterator implements Iterator<K> {
      
      private int position;
      private int lastNext = -1;
      
      @Override
      public boolean hasNext() {
        goToNext();
        return position < keys.length;
      }
      
      @Override
      public K next() {
        goToNext();
        lastNext = position;
        if (position >= keys.length) {
          throw new NoSuchElementException();
        }
        return keys[position++];
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
      return containsValue(o);
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
    
    private final class ValueIterator implements Iterator<V> {
      
      private int position;
      private int lastNext = -1;
      
      @Override
      public boolean hasNext() {
        goToNext();
        return position < values.length;
      }
      
      @Override
      public V next() {
        goToNext();
        lastNext = position;
        if (position >= values.length) {
          throw new NoSuchElementException();
        }
        return values[position++];
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
