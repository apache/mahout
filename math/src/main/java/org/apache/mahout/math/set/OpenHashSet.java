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

package org.apache.mahout.math.set;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.mahout.math.MurmurHash;
import org.apache.mahout.math.function.ObjectProcedure;
import org.apache.mahout.math.map.PrimeFinder;

/**
  * Open hashing alternative to java.util.HashSet.
 **/
public class OpenHashSet<T> extends AbstractSet implements Set<T>  {
  protected static final byte FREE = 0;
  protected static final byte FULL = 1;
  protected static final byte REMOVED = 2;
  protected static final char NO_KEY_VALUE = 0;

  /** The hash table keys. */
  private Object[] table;

  /** The state of each hash table entry (FREE, FULL, REMOVED). */
  private byte[] state;

  /** The number of table entries in state==FREE. */
  private int freeEntries;


  /** Constructs an empty map with default capacity and default load factors. */
  public OpenHashSet() {
    this(DEFAULT_CAPACITY);
  }

  /**
   * Constructs an empty map with the specified initial capacity and default load factors.
   *
   * @param initialCapacity the initial capacity of the map.
   * @throws IllegalArgumentException if the initial capacity is less than zero.
   */
  public OpenHashSet(int initialCapacity) {
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
  public OpenHashSet(int initialCapacity, double minLoadFactor, double maxLoadFactor) {
    setUp(initialCapacity, minLoadFactor, maxLoadFactor);
  }

  /** Removes all values associations from the receiver. Implicitly calls <tt>trimToSize()</tt>. */
  @Override
  public void clear() {
    Arrays.fill(this.state, 0, state.length - 1, FREE);
    distinct = 0;
    freeEntries = table.length; // delta
    trimToSize();
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @SuppressWarnings("unchecked")
  @Override
  public Object clone() {
    OpenHashSet<T> copy = (OpenHashSet<T>) super.clone();
    copy.table = copy.table.clone();
    copy.state = copy.state.clone();
    return copy;
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  @Override
  @SuppressWarnings("unchecked")
  public boolean contains(Object key) {
    return indexOfKey((T)key) >= 0;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of associations without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver. <p> This
   * method never need be called; it is for performance tuning only. Calling this method before <tt>add()</tt>ing a
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
  public boolean forEachKey(ObjectProcedure<T> procedure) {
    for (int i = table.length; i-- > 0;) {
      if (state[i] == FULL) {
        if (!procedure.apply((T)table[i])) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @param key the key to be added to the receiver.
   * @return the index where the key would need to be inserted, if it is not already contained. Returns -index-1 if the
   *         key is already contained at slot index. Therefore, if the returned index < 0, then it is already contained
   *         at slot -index-1. If the returned index >= 0, then it is NOT already contained and should be inserted at
   *         slot index.
   */
  protected int indexOfInsertion(T key) {
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
    while (stat[i] == FULL && tab[i] != key) {
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
  protected int indexOfKey(T key) {
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
    while (stat[i] != FREE && (stat[i] == REMOVED || (!key.equals(tab[i])))) {
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
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. 
   * This method can be used
   * to iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  @SuppressWarnings("unchecked")
  public void keys(List<T> list) {
    list.clear();
  

    Object [] tab = table;
    byte[] stat = state;

    for (int i = tab.length; i-- > 0;) {
      if (stat[i] == FULL) {
        list.add((T)tab[i]);
      }
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public boolean add(Object key) {
    int i = indexOfInsertion((T)key);
    if (i < 0) { //already contained
      return false;
    }

    if (this.distinct > this.highWaterMark) {
      int newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
      return add(key);
    }

    this.table[i] = key;
    if (this.state[i] == FREE) {
      this.freeEntries--;
    }
    this.state[i] = FULL;
    this.distinct++;

    if (this.freeEntries < 1) { //delta
      int newCapacity = chooseGrowCapacity(this.distinct + 1, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
      return add(key);
    }
    
    return true;
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
    byte[] oldState = state;

    Object[] newTable = new Object[newCapacity];
    byte[] newState = new byte[newCapacity];

    this.lowWaterMark = chooseLowWaterMark(newCapacity, this.minLoadFactor);
    this.highWaterMark = chooseHighWaterMark(newCapacity, this.maxLoadFactor);

    this.table = newTable;
    this.state = newState;
    this.freeEntries = newCapacity - this.distinct; // delta

    for (int i = oldCapacity; i-- > 0;) {
      if (oldState[i] == FULL) {
        Object element = oldTable[i];
        int index = indexOfInsertion((T)element);
        newTable[index] = element;
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
  public boolean remove(Object key) {
    int i = indexOfKey((T)key);
    if (i < 0) {
      return false;
    } // key not contained

    this.state[i] = REMOVED;
    this.distinct--;

    if (this.distinct < this.lowWaterMark) {
      int newCapacity = chooseShrinkCapacity(this.distinct, this.minLoadFactor, this.maxLoadFactor);
      rehash(newCapacity);
    }

    return true;
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

  @Override
  public boolean isEmpty() {
    return size() == 0;
  }
  
  /**
   * OpenHashSet instances are only equal to other OpenHashSet instances, not to 
   * any other collection. Hypothetically, we should check for and permit
   * equals on other Sets.
   */
  @Override
  @SuppressWarnings("unchecked")
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof OpenHashSet)) {
      return false;
    }
    final OpenHashSet<T> other = (OpenHashSet<T>) obj;
    if (other.size() != size()) {
      return false;
    }

    return forEachKey(new ObjectProcedure<T>() {
      @Override
      public boolean apply(T key) {
        return other.contains(key);
      }
    });
  }

  @Override
  public int hashCode() {
    ByteBuffer buf = ByteBuffer.allocate(size());
    for (int i = 0; i < table.length; i++) {
      Object v = table[i];
      if (state[i] == FULL) {
        buf.putInt(v.hashCode());
      }
    }
    return MurmurHash.hash(buf, this.getClass().getName().hashCode());
  }

  /**
   * Implement the standard Java Collections iterator. Note that 'remove' is silently
   * ineffectual here. This method is provided for convenience, only.
   */
  @Override
  public Iterator<T> iterator() {
    List<T> keyList = new ArrayList<T>();
    keys(keyList);
    return keyList.iterator();
  }

  @Override
  public Object[] toArray() {
    List<T> keyList = new ArrayList<T>();
    keys(keyList);
    return keyList.toArray();
  }

  @Override
  public boolean addAll(Collection<? extends T> c) {
    boolean anyAdded = false;
    for (T o : c) {
      boolean added = add(o);
      anyAdded |= added;
    }
    return anyAdded;
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    for (Object o : c) {
      if (!contains(o)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    boolean anyRemoved = false;
    for (Object o : c) {
      boolean removed = remove(o);
      anyRemoved |= removed;
    }
    return anyRemoved;
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    final Collection<?> finalCollection = c;
    final boolean[] modified = new boolean[1];
    modified[0] = false;
    forEachKey(new ObjectProcedure<T>() {
      @Override
      public boolean apply(T element) {
        if (!finalCollection.contains(element)) {
          remove(element);
          modified[0] = true;
        }
        return true;
      }
    });
    return modified[0];
  }

  @Override
  public <T> T[] toArray(T[] a) {
    return keys().toArray(a);
  }

  public List<T> keys() {
    List<T> keys = new ArrayList<T>();
    keys(keys);
    return keys;
  }
}
