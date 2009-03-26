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

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Set;
import java.lang.reflect.Array;
import java.io.Serializable;

/**
 * <p>This is an optimized {@link Set} implementation, based on algorithms described in Knuth's
 * "Art of Computer Programming", Vol. 3, p. 529.</p>
 *
 * <p>It should be faster than {@link java.util.HashSet} in some cases, but not all. It should definitely
 * be more memory efficient since that implementation is actually just a {@link java.util.HashMap} underneath
 * mapping values to a dummy object.</p>
 *
 * <p>This class is not a bit thread-safe.</p>
 *
 * <p>This implementation does not allow <code>null</code> as a key.</p>
 *
 * @see FastMap
 */
public final class FastSet<K> implements Set<K>, Serializable, Cloneable {

  /**
   * Dummy object used to represent a key that has been removed.
   */
  private static final Object REMOVED = new Object();

  private K[] keys;
  private int numEntries;
  private int numSlotsUsed;

  /**
   * Creates a new {@link FastSet} with default capacity.
   */
  public FastSet() {
    this(5);
  }

  public FastSet(Collection<? extends K> c) {
    this(c.size());
    addAll(c);
  }

  @SuppressWarnings("unchecked")  
  public FastSet(int size) {
    if (size < 1) {
      throw new IllegalArgumentException("size must be at least 1");
    }
    if (size >= RandomUtils.MAX_INT_SMALLER_TWIN_PRIME >> 1) {
      throw new IllegalArgumentException("size must be less than " + (RandomUtils.MAX_INT_SMALLER_TWIN_PRIME >> 1));
    }
    int hashSize = RandomUtils.nextTwinPrime(2 * size);
    keys = (K[]) new Object[hashSize];
  }

  /**
   * This is for the benefit of inner classes. Without it the compiler would just generate a similar synthetic
   * accessor. Might as well make it explicit.
   */
  K[] getKeys() {
    return keys;
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

  @Override
  public int size() {
    return numEntries;
  }

  @Override
  public boolean isEmpty() {
    return numEntries == 0;
  }

  @Override
  public boolean contains(Object key) {
    return key != null && keys[find(key)] != null;
  }

  /**
   * @throws NullPointerException if key is null
   */
  @Override
  public boolean add(K key) {
    if (key == null) {
      throw new NullPointerException();
    }
    // If less than half the slots are open, let's clear it up
    if (numSlotsUsed >= keys.length >> 1) {
      // If over half the slots used are actual entries, let's grow
      if (numEntries >= numSlotsUsed >> 1) {
        growAndRehash();
      } else {
        // Otherwise just rehash to clear REMOVED entries and don't grow
        rehash();
      }
    }
    // Here we may later consider implementing Brent's variation described on page 532
    int index = find(key);
    if (keys[index] == null) {
      keys[index] = key;
      numEntries++;
      numSlotsUsed++;
      return true;
    }
    return false;
  }

  @Override
  public Iterator<K> iterator() {
    return new KeyIterator();
  }

  @Override
  public boolean remove(Object key) {
    if (key == null) {
      return false;
    }
    int index = find(key);
    if (keys[index] == null) {
      return false;
    } else {
      ((Object[]) keys)[index] = REMOVED;
      numEntries--;
      return true;
    }
    // Could un-set recentlyAccessed's bit but doesn't matter
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    for (Object o : c) {
      if (o == null || keys[find(o)] == null) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean addAll(Collection<? extends K> c) {
    boolean changed = false;
    for (K k : c) {
      if (add(k)) {
        changed = true;
      }
    }
    return changed;
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    boolean changed = false;
    Iterator<K> iterator = iterator();
    while (iterator.hasNext()) {
      K k = iterator.next();
      if (!c.contains(k)) {
        iterator.remove();
        changed = true;
      }
    }
    return changed;
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    boolean changed = false;
    for (Object o : c) {
      if (remove(o)) {
        changed = true;
      }
    }
    return changed;
  }

  @Override
  public void clear() {
    numEntries = 0;
    numSlotsUsed = 0;
    Arrays.fill(keys, null);
  }

  @Override
  public Object[] toArray() {
    return toArray(new Object[numEntries]);
  }

  @Override
  @SuppressWarnings("unchecked")
  public <T> T[] toArray(T[] a) {
    if (a.length < numEntries) {
      a = (T[]) Array.newInstance(a.getClass().getComponentType(), numEntries);
    }
    int keyOffset = 0;
    int resultOffset = 0;
    while (resultOffset < a.length) {
      K key = keys[keyOffset++];
      if (key != null && key != REMOVED) {
        a[resultOffset++] = (T) key;
      }
    }
    return a;
  }

  private void growAndRehash() {
    if (keys.length >= RandomUtils.MAX_INT_SMALLER_TWIN_PRIME >> 1) {
      throw new IllegalStateException("Can't grow any more");
    }
    rehash(RandomUtils.nextTwinPrime(keys.length << 1));
  }

  public void rehash() {
    rehash(RandomUtils.nextTwinPrime(numEntries << 1));
  }

  @SuppressWarnings("unchecked")
  private void rehash(int newHashSize) {
    K[] oldKeys = keys;
    numEntries = 0;
    numSlotsUsed = 0;
    keys = (K[]) new Object[newHashSize];
    int length = oldKeys.length;
    for (int i = 0; i < length; i++) {
      K key = oldKeys[i];
      if (key != null && key != REMOVED) {
        add(key);
      }
    }
  }

  /**
   * Convenience method to quickly compute just the size of the intersection with another {@link FastSet}.
   */
  @SuppressWarnings("unchecked")  
  public int intersectionSize(FastSet<?> other) {
    int count = 0;
    K[] otherKeys = (K[]) other.keys;
    for (K key : otherKeys) {
      if (key != null && key != REMOVED && keys[find(key)] != null) {
        count++;
      }
    }
    return count;
  }

  @Override
  public FastSet<K> clone() {
    FastSet<K> clone;
    try {
      clone = (FastSet<K>) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new AssertionError();
    }
    clone.keys = (K[]) new Object[keys.length];
    System.arraycopy(keys, 0, clone.keys, 0, keys.length);
    return clone;
  }

  private final class KeyIterator implements Iterator<K> {

    private int position;
    private int lastNext = -1;

    @Override
    public boolean hasNext() {
      goToNext();
      return position < getKeys().length;
    }

    @Override
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
      K[] keys = getKeys();
      int length = keys.length;
      while (position < length && (keys[position] == null || keys[position] == REMOVED)) {
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
      ((Object[]) keys)[lastNext] = REMOVED;
      numEntries--;
    }

  }


}