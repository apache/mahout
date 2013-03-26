/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
Copyright ï¿½ 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.map;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.mahout.math.function.ObjectObjectProcedure;
import org.apache.mahout.math.function.ObjectProcedure;
import org.apache.mahout.math.set.AbstractSet;
import org.apache.mahout.math.set.OpenHashSet;

/**
  * Open hash map. This implements Map, but it does not respect several aspects of the Map contract
  * that impose the very sorts of performance penalities that this class exists to avoid.
  * {@link #entrySet}, {@link #values}, and {@link #keySet()} do <strong>not</strong> return
  * collections that share storage with the main map, and changes to those returned objects
  * are <strong>not</strong> reflected in the container.
 **/
public class OpenHashMap<K,V> extends AbstractSet implements Map<K,V> {
  protected static final byte FREE = 0;
  protected static final byte FULL = 1;
  protected static final byte REMOVED = 2;
  protected static final Object NO_KEY_VALUE = null;

  /** The hash table keys. */
  protected Object[] table;

  /** The hash table values. */
  protected Object[] values;

  /** The state of each hash table entry (FREE, FULL, REMOVED). */
  protected byte[] state;

  /** The number of table entries in state==FREE. */
  protected int freeEntries;


  /** Constructs an empty map with default capacity and default load factors. */
  public OpenHashMap() {
    this(DEFAULT_CAPACITY);
  }

  /**
   * Constructs an empty map with the specified initial capacity and default load factors.
   *
   * @param initialCapacity the initial capacity of the map.
   * @throws IllegalArgumentException if the initial capacity is less than zero.
   */
  public OpenHashMap(int initialCapacity) {
    this(initialCapacity, DEFAULT_MIN_LOAD_FACTOR, DEFAULT_MAX_LOAD_FACTOR);
  }

  /**
   * Constructs an empty map with the specified initial capacity and the specified minimum and maximum load factor.
   *
   * @param initialCapacity the initial capacity.
   * @param minLoadFactor   the minimum load factor.
   * @param maxLoadFactor   the maximum load factor.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   */
  public OpenHashMap(int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    setUp(initialCapacity, minLoadFactor, maxLoadFactor);
  }

  /** Removes all (key,value) associations from the receiver. Implicitly calls <tt>trimToSize()</tt>. */
  @Override
  public void clear() {
    Arrays.fill(this.state, FREE);
    distinct = 0;
    freeEntries = table.length; // delta
    trimToSize();
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  @SuppressWarnings("unchecked")
  public Object clone() {
    OpenHashMap<K,V> copy = (OpenHashMap<K,V>) super.clone();
    copy.table = copy.table.clone();
    copy.values = copy.values.clone();
    copy.state = copy.state.clone();
    return copy;
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  @SuppressWarnings("unchecked")
  @Override
  public boolean containsKey(Object key) {
    return indexOfKey((K)key) >= 0;
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified value.
   *
   * @return <tt>true</tt> if the receiver contains the specified value.
   */
  @SuppressWarnings("unchecked")
  @Override
  public boolean containsValue(Object value) {
    return indexOfValue((V)value) >= 0;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of associations without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver. <p> This
   * method never need be called; it is for performance tuning only. Calling this method before <tt>put()</tt>ing a
   * large number of associations boosts performance, because the receiver will grow only once instead of potentially
   * many times and hash collisions get less probable.
   *
   * @param minCapacity the desired minimum capacity.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    if (table.length < minCapacity) {
      int newCapacity = nextPrime(minCapacity);
      rehash(newCapacity);
    }
  }

  /**
   * Applies a procedure to each key of the receiver, if any. Note: Iterates over the keys in no particular order.
   * Subclasses can define a particular order, for example, "sorted by key". All methods which <i>can</i> be expressed
   * in terms of this method (most methods can) <i>must guarantee</i> to use the <i>same</i> order defined by this
   * method, even if it is no particular order. This is necessary so that, for example, methods <tt>keys</tt> and
   * <tt>values</tt> will yield association pairs, not two uncorrelated lists.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  @SuppressWarnings("unchecked")
  public boolean forEachKey(ObjectProcedure<K> procedure) {
    for (int i = table.length; i-- > 0;) {
      if (state[i] == FULL && !procedure.apply((K)table[i])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Applies a procedure to each (key,value) pair of the receiver, if any. Iteration order is guaranteed to be
   * <i>identical</i> to the order used by method {@link #forEachKey(ObjectProcedure)}.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  @SuppressWarnings("unchecked")
  public boolean forEachPair(ObjectObjectProcedure<K,V> procedure) {
    for (int i = table.length; i-- > 0;) {
      if (state[i] == FULL && !procedure.apply((K)table[i], (V)values[i])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the value associated with the specified key. It is often a good idea to first check with {@link
   * #containsKey(Object)} whether the given key has a value associated or not, i.e. whether there exists an association
   * for the given key or not.
   *
   * @param key the key to be searched for.
   * @return the value associated with the specified key; <tt>0</tt> if no such key is present.
   */
  @SuppressWarnings("unchecked")
  @Override
  public V get(Object key) {
    int i = indexOfKey((K)key);
    if (i < 0) {
      return null;
    } //not contained
    return (V)values[i];
  }

  /**
   * @param key the key to be added to the receiver.
   * @return the index where the key would need to be inserted, if it is not already contained. Returns -index-1 if the
   *         key is already contained at slot index. Therefore, if the returned index < 0, then it is already contained
   *         at slot -index-1. If the returned index >= 0, then it is NOT already contained and should be inserted at
   *         slot index.
   */
  protected int indexOfInsertion(K key) {
    Object[] tab = table;
    byte[] stat = state;
    int length = tab.length;

    int hash = key.hashCode() & 0x7FFFFFFF;
    int i = hash % length;
    int decrement = hash % (length - 2); // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //int decrement = (hash / length) % length;
    if (decrement == 0) {
      decrement = 1;
    }

    // stop if we find a removed or free slot, or if we find the key itself
    // do NOT skip over removed slots (yes, open addressing is like that...)
    while (stat[i] == FULL && !equalsMindTheNull(key, tab[i])) {
      i -= decrement;
      //hashCollisions++;
      if (i < 0) {
        i += length;
      }
    }

    if (stat[i] == REMOVED) {
      // stop if we find a free slot, or if we find the key itself.
      // do skip over removed slots (yes, open addressing is like that...)
      // assertion: there is at least one FREE slot.
      int j = i;
      while (stat[i] != FREE && (stat[i] == REMOVED || tab[i] != key)) {
        i -= decrement;
        //hashCollisions++;
        if (i < 0) {
          i += length;
        }
      }
      if (stat[i] == FREE) {
        i = j;
      }
    }


    if (stat[i] == FULL) {
      // key already contained at slot i.
      // return a negative number identifying the slot.
      return -i - 1;
    }
    // not already contained, should be inserted at slot i.
    // return a number >= 0 identifying the slot.
    return i;
  }

  /**
   * @param key the key to be searched in the receiver.
   * @return the index where the key is contained in the receiver, returns -1 if the key was not found.
   */
  protected int indexOfKey(K key) {
    Object[] tab = table;
    byte[] stat = state;
    int length = tab.length;

    int hash = key.hashCode() & 0x7FFFFFFF;
    int i = hash % length;
    int decrement = hash % (length - 2); // double hashing, see http://www.eece.unm.edu/faculty/heileman/hash/node4.html
    //int decrement = (hash / length) % length;
    if (decrement == 0) {
      decrement = 1;
    }

    // stop if we find a free slot, or if we find the key itself.
    // do skip over removed slots (yes, open addressing is like that...)
    while (stat[i] != FREE && (stat[i] == REMOVED || !equalsMindTheNull(key, tab[i]))) {
      i -= decrement;
      //hashCollisions++;
      if (i < 0) {
        i += length;
      }
    }

    if (stat[i] == FREE) {
      return -1;
    } // not found
    return i; //found, return index where key is contained
  }

  /**
   * @param value the value to be searched in the receiver.
   * @return the index where the value is contained in the receiver, returns -1 if the value was not found.
   */
  protected int indexOfValue(V value) {
    Object[] val = values;
    byte[] stat = state;

    for (int i = stat.length; --i >= 0;) {
      if (stat[i] == FULL && equalsMindTheNull(val[i], value)) {
        return i;
      }
    }

    return -1; // not found
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. 
   * This method can be used
   * to iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  @SuppressWarnings("unchecked")
  public void keys(List<K> list) {
    list.clear();
  

    Object [] tab = table;
    byte[] stat = state;

    for (int i = tab.length; i-- > 0;) {
      if (stat[i] == FULL) {
        list.add((K)tab[i]);
      }
    }
  }

  /**
   * Associates the given key with the given value. Replaces any old <tt>(key,someOtherValue)</tt> association, if
   * existing.
   *
   * @param key   the key the value shall be associated with.
   * @param value the value to be associated.
   * @return <tt>true</tt> if the receiver did not already contain such a key; <tt>false</tt> if the receiver did
   *         already contain such a key - the new value has now replaced the formerly associated value.
   */
  @SuppressWarnings("unchecked")
  @Override
  public V put(K key, V value) {
    int i = indexOfInsertion(key);
    if (i < 0) { //already contained
      i = -i - 1;
      V previous = (V) this.values[i];
      this.values[i] = value;
      return previous;
    }

    if (this.distinct > this.highWaterMark) {
      int newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
      return put(key, value);
    }

    this.table[i] = key;
    this.values[i] = value;
    if (this.state[i] == FREE) {
      this.freeEntries--;
    }
    this.state[i] = FULL;
    this.distinct++;

    if (this.freeEntries < 1) { //delta
      int newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
    }

    return null;
  }

  /**
   * Rehashes the contents of the receiver into a new table with a smaller or larger capacity. This method is called
   * automatically when the number of keys in the receiver exceeds the high water mark or falls below the low water
   * mark.
   */
  @SuppressWarnings("unchecked")
  protected void rehash(int newCapacity) {
    int oldCapacity = table.length;
    //if (oldCapacity == newCapacity) return;

    Object[] oldTable = table;
    Object[] oldValues = values;
    byte[] oldState = state;

    Object[] newTable = new Object[newCapacity];
    Object[] newValues = new Object[newCapacity];
    byte[] newState = new byte[newCapacity];

    this.lowWaterMark = chooseLowWaterMark(newCapacity, this.minLoadFactor);
    this.highWaterMark = chooseHighWaterMark(newCapacity, this.maxLoadFactor);

    this.table = newTable;
    this.values = newValues;
    this.state = newState;
    this.freeEntries = newCapacity - this.distinct; // delta

    for (int i = oldCapacity; i-- > 0;) {
      if (oldState[i] == FULL) {
        Object element = oldTable[i];
        int index = indexOfInsertion((K)element);
        newTable[index] = element;
        newValues[index] = oldValues[i];
        newState[index] = FULL;
      }
    }
  }

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  @SuppressWarnings("unchecked")
  @Override
  public V remove(Object key) {
    int i = indexOfKey((K)key);
    if (i < 0) {
      return null;
    }
    // key not contained
    V removed = (V) values[i];

    this.state[i] = REMOVED;
    //this.values[i]=0; // delta
    this.distinct--;

    if (this.distinct < this.lowWaterMark) {
      int newCapacity = chooseShrinkCapacity(this.distinct, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
    }

    return removed;
  }

  /**
   * Initializes the receiver.
   *
   * @param initialCapacity the initial capacity of the receiver.
   * @param minLoadFactor   the minLoadFactor of the receiver.
   * @param maxLoadFactor   the maxLoadFactor of the receiver.
   * @throws IllegalArgumentException if <tt>initialCapacity < 0 || (minLoadFactor < 0.0 || minLoadFactor >= 1.0) ||
   *                                  (maxLoadFactor <= 0.0 || maxLoadFactor >= 1.0) || (minLoadFactor >=
   *                                  maxLoadFactor)</tt>.
   */
  @Override
  protected void setUp(int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    int capacity = initialCapacity;
    super.setUp(capacity, minLoadFactor, maxLoadFactor);
    capacity = nextPrime(capacity);
    if (capacity == 0) {
      capacity = 1;
    } // open addressing needs at least one FREE slot at any time.

    this.table = new Object[capacity];
    this.values = new Object[capacity];
    this.state = new byte[capacity];

    // memory will be exhausted long before this pathological case happens, anyway.
    this.minLoadFactor = minLoadFactor;
    if (capacity == PrimeFinder.LARGEST_PRIME) {
      this.maxLoadFactor = 1.0;
    } else {
      this.maxLoadFactor = maxLoadFactor;
    }

    this.distinct = 0;
    this.freeEntries = capacity; // delta

    // lowWaterMark will be established upon first expansion.
    // establishing it now (upon instance construction) would immediately make the table shrink upon first put(...).
    // After all the idea of an "initialCapacity" implies violating lowWaterMarks when an object is young.
    // See ensureCapacity(...)
    this.lowWaterMark = 0;
    this.highWaterMark = chooseHighWaterMark(capacity, this.maxLoadFactor);
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. Releases any superfluous internal memory. An
   * application can use this operation to minimize the storage of the receiver.
   */
  @Override
  public void trimToSize() {
    // * 1.2 because open addressing's performance exponentially degrades beyond that point
    // so that even rehashing the table can take very long
    int newCapacity = nextPrime((int) (1 + 1.2 * size()));
    if (table.length > newCapacity) {
      rehash(newCapacity);
    }
  }

  /**
   * Access for unit tests.
   * @param capacity
   * @param minLoadFactor
   * @param maxLoadFactor
   */
  void getInternalFactors(int[] capacity, 
      double[] minLoadFactor, 
      double[] maxLoadFactor) {
    capacity[0] = table.length;
    minLoadFactor[0] = this.minLoadFactor;
    maxLoadFactor[0] = this.maxLoadFactor;
  }

  private class MapEntry implements Map.Entry<K,V> {
    private final K key;
    private final V value;
    
    MapEntry(K key, V value) {
      this.key = key;
      this.value = value;
    }

    @Override
    public K getKey() {
      return key;
    }

    @Override
    public V getValue() {
      return value;
    }

    @Override
    public V setValue(V value) {
      throw new UnsupportedOperationException("Map.Entry.setValue not supported for OpenHashMap");
    }
    
  }

  /**
   * Allocate a set to contain Map.Entry objects for the pairs and return it.
   */
  @Override
  public Set<java.util.Map.Entry<K,V>> entrySet() {
    final Set<Entry<K, V>> entries = new OpenHashSet<Map.Entry<K,V>>();
    forEachPair(new ObjectObjectProcedure<K,V>() {
      @Override
      public boolean apply(K key, V value) {
        entries.add(new MapEntry(key, value));
        return true;
      }
    });
    return entries;
  }

  /**
   * Allocate a set to contain keys and return it.
   * This violates the 'backing' provisions of the map interface.
   */
  @Override
  public Set<K> keySet() {
    final Set<K> keys = new OpenHashSet<K>();
    forEachKey(new ObjectProcedure<K>() {
      @Override
      public boolean apply(K element) {
        keys.add(element);
        return true;
      }
    });
    return keys;
  }

  @Override
  public void putAll(Map<? extends K,? extends V> m) {
    for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
      put(e.getKey(), e.getValue());
    }
  }

  /**
   * Allocate a list to contain the values and return it.
   * This violates the 'backing' provision of the Map interface.
   */
  @Override
  public Collection<V> values() {
    final List<V> valueList = new ArrayList<V>();
    forEachPair(new ObjectObjectProcedure<K,V>() {
      @Override
      public boolean apply(K key, V value) {
        valueList.add(value);
        return true;
      }
    });
    return valueList;
  }

  @SuppressWarnings("unchecked")
  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof OpenHashMap)) {
      return false;
    }
    final OpenHashMap<K,V> o = (OpenHashMap<K,V>) obj;
    if (o.size() != size()) {
      return false;
    }
    final boolean[] equal = new boolean[1];
    equal[0] = true;
    forEachPair(new ObjectObjectProcedure<K,V>() {
      @Override
      public boolean apply(K key, V value) {
        Object ov = o.get(key);
        if (!value.equals(ov)) {
          equal[0] = false;
          return false;
        }
        return true;
      }
    });
    return equal[0];
  }

  @Override
  public String toString() {
    final StringBuilder sb = new StringBuilder();
    sb.append('{');
    forEachPair(new ObjectObjectProcedure<K,V>() {
      @Override
      public boolean apply(K key, V value) {
        sb.append('[');
        sb.append(key);
        sb.append(" -> ");
        sb.append(value);
        sb.append("] ");
        return true;
      }
    });
    sb.append('}');
    return sb.toString();
  }
}
