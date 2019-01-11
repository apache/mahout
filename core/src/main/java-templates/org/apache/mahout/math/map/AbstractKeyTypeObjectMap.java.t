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
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.map;

import java.util.ArrayList;
import java.util.List;
import java.nio.IntBuffer;
import java.util.Arrays;

import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.set.HashUtils;
import org.apache.mahout.math.function.IntComparator;

import org.apache.mahout.math.function.${keyTypeCap}ObjectProcedure;
import org.apache.mahout.math.function.${keyTypeCap}Procedure;
import org.apache.mahout.math.list.${keyTypeCap}ArrayList;
import org.apache.mahout.math.set.AbstractSet;

public abstract class Abstract${keyTypeCap}ObjectMap<T> extends AbstractSet {

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  public boolean containsKey(final ${keyType} key) {
    return !forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} iterKey) {
            return (key != iterKey);
          }
        }
    );
  }

  /**
   * Returns <tt>true</tt> if the receiver contains the specified value. Tests for identity.
   *
   * @return <tt>true</tt> if the receiver contains the specified value.
   */
  public boolean containsValue(final T value) {
    return !forEachPair(
        new ${keyTypeCap}ObjectProcedure<T>() {
          @Override
          public boolean apply(${keyType} iterKey, Object iterValue) {
            return (value != iterValue);
          }
        }
    );
  }

  /**
   * Returns a deep copy of the receiver; uses <code>clone()</code> and casts the result.
   *
   * @return a deep copy of the receiver.
   */
  @SuppressWarnings("unchecked") // seemingly unavoidable.
  public Abstract${keyTypeCap}ObjectMap<T> copy() {
    return this.getClass().cast(clone());
  }

  /**
   * Compares the specified object with this map for equality.  Returns <tt>true</tt> if the given object is also a map
   * and the two maps represent the same mappings.  More formally, two maps <tt>m1</tt> and <tt>m2</tt> represent the
   * same mappings iff
   * <pre>
   * m1.forEachPair(
   *    new ${keyTypeCap}ObjectProcedure() {
   *      public boolean apply(${keyType} key, Object value) {
   *        return m2.containsKey(key) && m2.get(key) == value;
   *      }
   *    }
   *  )
   * &&
   * m2.forEachPair(
   *    new ${keyTypeCap}ObjectProcedure() {
   *      public boolean apply(${keyType} key, Object value) {
   *        return m1.containsKey(key) && m1.get(key) == value;
   *      }
   *    }
   *  );
   * </pre>
   *
   * This implementation first checks if the specified object is this map; if so it returns <tt>true</tt>.  Then, it
   * checks if the specified object is a map whose size is identical to the size of this set; if not, it it returns
   * <tt>false</tt>.  If so, it applies the iteration as described above.
   *
   * @param obj object to be compared for equality with this map.
   * @return <tt>true</tt> if the specified object is equal to this map.
   */
  @SuppressWarnings("unchecked") // incompressible
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof Abstract${keyTypeCap}ObjectMap)) {
      return false;
    }
    final Abstract${keyTypeCap}ObjectMap other = (Abstract${keyTypeCap}ObjectMap) obj;
    if (other.size() != size()) {
      return false;
    }

    return
        forEachPair(
            new ${keyTypeCap}ObjectProcedure() {
              @Override
              public boolean apply(${keyType} key, Object value) {
                return other.containsKey(key) && other.get(key) == value;
              }
            }
        )
            &&
            other.forEachPair(
                new ${keyTypeCap}ObjectProcedure() {
                  @Override
                  public boolean apply(${keyType} key, Object value) {
                    return containsKey(key) && get(key) == value;
                  }
                }
            );
  }

  public int hashCode() {
    final int[] buf = new int[size()];
    forEachPair(
      new ${keyTypeCap}ObjectProcedure() {
        int i = 0;

        @Override
        public boolean apply(${keyType} key, Object value) {
          buf[i++] = HashUtils.hash(key) ^ value.hashCode();
          return true;
        }
      }
    );
    Arrays.sort(buf);
    return IntBuffer.wrap(buf).hashCode();
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
  public abstract boolean forEachKey(${keyTypeCap}Procedure procedure);

  /**
   * Applies a procedure to each (key,value) pair of the receiver, if any. Iteration order is guaranteed to be
   * <i>identical</i> to the order used by method {@link #forEachKey(${keyTypeCap}Procedure)}.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  public boolean forEachPair(final ${keyTypeCap}ObjectProcedure<T> procedure) {
    return forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} key) {
            return procedure.apply(key, get(key));
          }
        }
    );
  }

  /**
   * Returns the value associated with the specified key. It is often a good idea to first check with {@link
   * #containsKey(${keyType})} whether the given key has a value associated or not, i.e. whether there exists an association
   * for the given key or not.
   *
   * @param key the key to be searched for.
   * @return the value associated with the specified key; <tt>null</tt> if no such key is present.
   */
  public abstract T get(${keyType} key);

  /**
   * Returns a list filled with all keys contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link
   * #forEachKey(${keyTypeCap}Procedure)}. <p> This method can be used to iterate over the keys of the receiver.
   *
   * @return the keys.
   */
  public ${keyTypeCap}ArrayList keys() {
    ${keyTypeCap}ArrayList list = new ${keyTypeCap}ArrayList(size());
    keys(list);
    return list;
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method {@link #forEachKey(${keyTypeCap}Procedure)}. <p> This method can be used to
   * iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  public void keys(final ${keyTypeCap}ArrayList list) {
    list.clear();
    forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} key) {
            list.add(key);
            return true;
          }
        }
    );
  }

  /**
   * Fills all keys <i>sorted ascending by their associated value</i> into the specified list. Fills into the list,
   * starting at index 0. After this call returns the specified list has a new size that equals <tt>this.size()</tt>.
   * Primary sort criterium is "value", secondary sort criterium is "key". This means that if any two values are equal,
   * the smaller key comes first. <p> <b>Example:</b> <br> <tt>keys = (8,7,6), values = (1,2,2) --> keyList =
   * (8,6,7)</tt>
   *
   * @param keyList the list to be filled, can have any size.
   */
  public void keysSortedByValue(${keyTypeCap}ArrayList keyList) {
    pairsSortedByValue(keyList, new ArrayList<T>(size()));
  }

  /**
   * Fills all pairs satisfying a given condition into the specified lists. Fills into the lists, starting at index 0.
   * After this call returns the specified lists both have a new size, the number of pairs satisfying the condition.
   * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(${keyTypeCap}Procedure)}.
   * <p> <b>Example:</b> <br>
   * <pre>
   * ${keyTypeCap}ObjectProcedure condition = new ${keyTypeCap}ObjectProcedure() { // match even keys only
   * public boolean apply(${keyType} key, Object value) { return key%2==0; }
   * }
   * keys = (8,7,6), values = (1,2,2) --> keyList = (6,8), valueList = (2,1)</tt>
   * </pre>
   *
   * @param condition the condition to be matched. Takes the current key as first and the current value as second
   *                  argument.
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  public void pairsMatching(final ${keyTypeCap}ObjectProcedure<T> condition, 
                            final ${keyTypeCap}ArrayList keyList,
                            final List<T> valueList) {
    keyList.clear();
    valueList.clear();

    forEachPair(
        new ${keyTypeCap}ObjectProcedure<T>() {
          @Override
          public boolean apply(${keyType} key, T value) {
            if (condition.apply(key, value)) {
              keyList.add(key);
              valueList.add(value);
            }
            return true;
          }
        }
    );
  }
  
  /**
   * Fills all keys and values <i>sorted ascending by key</i> into the specified lists. Fills into the lists, starting
   * at index 0. After this call returns the specified lists both have a new size that equals <tt>this.size()</tt>. <p>
   * <b>Example:</b> <br> <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (6,7,8), valueList = (2,2,1)</tt>
   *
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  @SuppressWarnings("unchecked")
  public void pairsSortedByKey(${keyTypeCap}ArrayList keyList, List<T> valueList) {
    keys(keyList);
    keyList.sort();
    // the following is straightforward if not the most space-efficient possibility
    Object[] tempValueList = new Object[keyList.size()];

    for (int i = keyList.size(); --i >= 0;) {
      tempValueList[i] = get(keyList.getQuick(i));
    }
    valueList.clear();
    for (Object value : tempValueList) {
      valueList.add((T) value);
    }
    
  }

  /**
   * Fills all keys and values <i>sorted ascending by value according to natural ordering</i> into the specified lists.
   * Fills into the lists, starting at index 0. After this call returns the specified lists both have a new size that
   * equals <tt>this.size()</tt>. Primary sort criterium is "value", secondary sort criterium is "key". This means that
   * if any two values are equal, the smaller key comes first. <p> <b>Example:</b> <br> <tt>keys = (8,7,6), values =
   * (1,2,2) --> keyList = (8,6,7), valueList = (1,2,2)</tt>
   *
   * @param keyList   the list to be filled with keys, can have any size.
   * @param valueList the list to be filled with values, can have any size.
   */
  @SuppressWarnings("unchecked")
  public void pairsSortedByValue(${keyTypeCap}ArrayList keyList, List<T> valueList) {
    keys(keyList);
    values(valueList);
    
    if (!valueList.isEmpty() && !(valueList.get(0) instanceof Comparable)) {
      throw new UnsupportedOperationException("Cannot sort the values; " 
          + valueList.get(0).getClass()
          + " does not implement Comparable");
    }

    final ${keyType}[] k = keyList.elements();
    final List<T> valueRef = valueList;
    Swapper swapper = new Swapper() {
      @Override
      public void swap(int a, int b) {
        T t1 = valueRef.get(a);
        valueRef.set(a, valueRef.get(b));
        valueRef.set(b, t1);
        ${keyType} t2 = k[a];
        k[a] = k[b];
        k[b] = t2;
      }
    };

    IntComparator comp = new IntComparator() {
      @Override
      public int compare(int a, int b) {
        int ab = ((Comparable)valueRef.get(a)).compareTo(valueRef.get(b));
        return ab < 0 ? -1 : ab > 0 ? 1 : (k[a] < k[b] ? -1 : (k[a] == k[b] ? 0 : 1));
      }
    };

    Sorting.quickSort(0, keyList.size(), comp, swapper);
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
  public abstract boolean put(${keyType} key, T value);

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  public abstract boolean removeKey(${keyType} key);

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by key.
   */
  public String toString() {
    ${keyTypeCap}ArrayList theKeys = keys();
    theKeys.sort();

    StringBuilder buf = new StringBuilder();
    buf.append('[');
    int maxIndex = theKeys.size() - 1;
    for (int i = 0; i <= maxIndex; i++) {
      ${keyType} key = theKeys.get(i);
      buf.append(String.valueOf(key));
      buf.append("->");
      buf.append(String.valueOf(get(key)));
      if (i < maxIndex) {
        buf.append(", ");
      }
    }
    buf.append(']');
    return buf.toString();
  }

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by value, according to natural ordering.
   */
  public String toStringByValue() {
    ${keyTypeCap}ArrayList theKeys = new ${keyTypeCap}ArrayList();
    keysSortedByValue(theKeys);

    StringBuilder buf = new StringBuilder();
    buf.append('[');
    int maxIndex = theKeys.size() - 1;
    for (int i = 0; i <= maxIndex; i++) {
      ${keyType} key = theKeys.get(i);
      buf.append(String.valueOf(key));
      buf.append("->");
      buf.append(String.valueOf(get(key)));
      if (i < maxIndex) {
        buf.append(", ");
      }
    }
    buf.append(']');
    return buf.toString();
  }

  /**
   * Returns a list filled with all values contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link
   * #forEachKey(${keyTypeCap}Procedure)}. <p> This method can be used to iterate over the values of the receiver.
   *
   * @return the values.
   */
  public List<T> values() {
    List<T> list = new ArrayList<T>(size());
    values(list);
    return list;
  }

  /**
   * Fills all values contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method {@link #forEachKey(${keyTypeCap}Procedure)}. <p> This method can be used to
   * iterate over the values of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  public void values(final List<T> list) {
    list.clear();
    forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} key) {
            list.add(get(key));
            return true;
          }
        }
    );
  }
}
