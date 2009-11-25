/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.map;

import org.apache.mahout.matrix.function.LongObjectProcedure;
import org.apache.mahout.matrix.function.LongProcedure;
import org.apache.mahout.matrix.list.LongArrayList;
import org.apache.mahout.matrix.list.ObjectArrayList;
/**
Abstract base class for hash maps holding (key,value) associations of type <tt>(long-->Object)</tt>.
First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
<p>
<b>Implementation</b>:
<p>
Almost all methods are expressed in terms of {@link #forEachKey(LongProcedure)}. 
As such they are fully functional, but inefficient. Override them in subclasses if necessary.

@author wolfgang.hoschek@cern.ch
@version 1.0, 09/24/99
@see      java.util.HashMap
*/
/** 
 * @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported.
 */
@Deprecated
public abstract class AbstractLongObjectMap extends AbstractMap {
  //public static int hashCollisions = 0; // for debug only
/**
 * Makes this class non instantiable, but still let's others inherit from it.
 */
protected AbstractLongObjectMap() {}
/**
 * Returns <tt>true</tt> if the receiver contains the specified key.
 *
 * @return <tt>true</tt> if the receiver contains the specified key.
 */
public boolean containsKey(final long key) {
  return ! forEachKey(
    new LongProcedure() {
      public boolean apply(long iterKey) {
        return (key != iterKey);
      }
    }
  );
}
/**
 * Returns <tt>true</tt> if the receiver contains the specified value.
 * Tests for identity.
 *
 * @return <tt>true</tt> if the receiver contains the specified value.
 */
public boolean containsValue(final Object value) {
  return ! forEachPair( 
    new LongObjectProcedure() {
      public boolean apply(long iterKey, Object iterValue) {
        return (value != iterValue);
      }
    }
  );
}
/**
 * Returns a deep copy of the receiver; uses <code>clone()</code> and casts the result.
 *
 * @return  a deep copy of the receiver.
 */
public AbstractLongObjectMap copy() {
  return (AbstractLongObjectMap) clone();
}
/**
 * Compares the specified object with this map for equality.  Returns
 * <tt>true</tt> if the given object is also a map and the two maps
 * represent the same mappings.  More formally, two maps <tt>m1</tt> and
 * <tt>m2</tt> represent the same mappings iff
 * <pre>
 * m1.forEachPair(
 *    new LongObjectProcedure() {
 *      public boolean apply(long key, Object value) {
 *        return m2.containsKey(key) && m2.get(key) == value;
 *      }
 *    }
 *  )
 * &&
 * m2.forEachPair(
 *    new LongObjectProcedure() {
 *      public boolean apply(long key, Object value) {
 *        return m1.containsKey(key) && m1.get(key) == value;
 *      }
 *    }
 *  );
 * </pre>
 *
 * This implementation first checks if the specified object is this map;
 * if so it returns <tt>true</tt>.  Then, it checks if the specified
 * object is a map whose size is identical to the size of this set; if
 * not, it it returns <tt>false</tt>.  If so, it applies the iteration as described above.
 *
 * @param obj object to be compared for equality with this map.
 * @return <tt>true</tt> if the specified object is equal to this map.
 */
public boolean equals(Object obj) {
  if (obj == this) return true;

  if (!(obj instanceof AbstractLongObjectMap)) return false;
  final AbstractLongObjectMap other = (AbstractLongObjectMap) obj;
  if (other.size() != size()) return false;

  return 
    forEachPair(
      new LongObjectProcedure() {
        public boolean apply(long key, Object value) {
          return other.containsKey(key) && other.get(key) == value;
        }
      }
    )
    &&
    other.forEachPair(
      new LongObjectProcedure() {
        public boolean apply(long key, Object value) {
          return containsKey(key) && get(key) == value;
        }
      }
    );
}
/**
 * Applies a procedure to each key of the receiver, if any.
 * Note: Iterates over the keys in no particular order.
 * Subclasses can define a particular order, for example, "sorted by key".
 * All methods which <i>can</i> be expressed in terms of this method (most methods can) <i>must guarantee</i> to use the <i>same</i> order defined by this method, even if it is no particular order.
 * This is necessary so that, for example, methods <tt>keys</tt> and <tt>values</tt> will yield association pairs, not two uncorrelated lists.
 *
 * @param procedure    the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise continues. 
 * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise. 
 */
public abstract boolean forEachKey(LongProcedure procedure);
/**
 * Applies a procedure to each (key,value) pair of the receiver, if any.
 * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 *
 * @param procedure    the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise continues. 
 * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise. 
 */
public boolean forEachPair(final LongObjectProcedure procedure) {
  return forEachKey(
    new LongProcedure() {
      public boolean apply(long key) {
        return procedure.apply(key,get(key));
      }
    }
  );
}
/**
 * Returns the value associated with the specified key.
 * It is often a good idea to first check with {@link #containsKey(long)} whether the given key has a value associated or not, i.e. whether there exists an association for the given key or not.
 *
 * @param key the key to be searched for.
 * @return the value associated with the specified key; <tt>null</tt> if no such key is present.
 */
public abstract Object get(long key);
/**
 * Returns the first key the given value is associated with.
 * It is often a good idea to first check with {@link #containsValue(Object)} whether there exists an association from a key to this value.
 * Search order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 *
 * @param value the value to search for.
 * @return the first key for which holds <tt>get(key) == value</tt>; 
 *       returns <tt>Long.MIN_VALUE</tt> if no such key exists.
 */
public long keyOf(final Object value) {
  final long[] foundKey = new long[1];
  boolean notFound = forEachPair(
    new LongObjectProcedure() {
      public boolean apply(long iterKey, Object iterValue) {
        boolean found = value == iterValue;
        if (found) foundKey[0] = iterKey;
        return !found;
      }
    }
  );
  if (notFound) return Long.MIN_VALUE;
  return foundKey[0];
}
/**
 * Returns a list filled with all keys contained in the receiver.
 * The returned list has a size that equals <tt>this.size()</tt>.
 * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 * <p>
 * This method can be used to iterate over the keys of the receiver.
 *
 * @return the keys.
 */
public LongArrayList keys() {
  LongArrayList list = new LongArrayList(size());
  keys(list);
  return list;
}
/**
 * Fills all keys contained in the receiver into the specified list.
 * Fills the list, starting at index 0.
 * After this call returns the specified list has a new size that equals <tt>this.size()</tt>.
 * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 * <p>
 * This method can be used to iterate over the keys of the receiver.
 *
 * @param list the list to be filled, can have any size.
 */
public void keys(final LongArrayList list) {
  list.clear();
  forEachKey(
    new LongProcedure() {
      public boolean apply(long key) {
        list.add(key);
        return true;
      }
    }
  );
}
/**
 * Fills all keys <i>sorted ascending by their associated value</i> into the specified list.
 * Fills into the list, starting at index 0.
 * After this call returns the specified list has a new size that equals <tt>this.size()</tt>.
 * Primary sort criterium is "value", secondary sort criterium is "key". 
 * This means that if any two values are equal, the smaller key comes first.
 * <p>
 * <b>Example:</b>
 * <br>
 * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (8,6,7)</tt>
 *
 * @param keyList the list to be filled, can have any size.
 */
public void keysSortedByValue(final LongArrayList keyList) {
  pairsSortedByValue(keyList, new ObjectArrayList(size()));
}
/**
Fills all pairs satisfying a given condition into the specified lists.
Fills into the lists, starting at index 0.
After this call returns the specified lists both have a new size, the number of pairs satisfying the condition.
Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
<p>
<b>Example:</b>
<br>
<pre>
LongObjectProcedure condition = new LongObjectProcedure() { // match even keys only
  public boolean apply(long key, Object value) { return key%2==0; }
}
keys = (8,7,6), values = (1,2,2) --> keyList = (6,8), valueList = (2,1)</tt>
</pre>

@param condition    the condition to be matched. Takes the current key as first and the current value as second argument.
@param keyList the list to be filled with keys, can have any size.
@param valueList the list to be filled with values, can have any size.
*/
public void pairsMatching(final LongObjectProcedure condition, final LongArrayList keyList, final ObjectArrayList valueList) {
  keyList.clear();
  valueList.clear();
  
  forEachPair(
    new LongObjectProcedure() {
      public boolean apply(long key, Object value) {
        if (condition.apply(key,value)) {
          keyList.add(key);
          valueList.add(value);
        }
        return true;
      }
    }
  );
}
/**
 * Fills all keys and values <i>sorted ascending by key</i> into the specified lists.
 * Fills into the lists, starting at index 0.
 * After this call returns the specified lists both have a new size that equals <tt>this.size()</tt>.
 * <p>
 * <b>Example:</b>
 * <br>
 * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (6,7,8), valueList = (2,2,1)</tt>
 *
 * @param keyList the list to be filled with keys, can have any size.
 * @param valueList the list to be filled with values, can have any size.
 */
public void pairsSortedByKey(final LongArrayList keyList, final ObjectArrayList valueList) {
  keys(keyList);
  keyList.sort();
  valueList.setSize(keyList.size());
  for (int i=keyList.size(); --i >= 0; ) {
    valueList.setQuick(i,get(keyList.getQuick(i)));
  }
}
/**
 * Fills all keys and values <i>sorted ascending by value according to natural ordering</i> into the specified lists.
 * Fills into the lists, starting at index 0.
 * After this call returns the specified lists both have a new size that equals <tt>this.size()</tt>.
 * Primary sort criterium is "value", secondary sort criterium is "key". 
 * This means that if any two values are equal, the smaller key comes first.
 * <p>
 * <b>Example:</b>
 * <br>
 * <tt>keys = (8,7,6), values = (1,2,2) --> keyList = (8,6,7), valueList = (1,2,2)</tt>
 *
 * @param keyList the list to be filled with keys, can have any size.
 * @param valueList the list to be filled with values, can have any size.
 */
public void pairsSortedByValue(final LongArrayList keyList, final ObjectArrayList valueList) {
  keys(keyList);
  values(valueList);
  
  final long[] k = keyList.elements();
  final Object[] v = valueList.elements();
  org.apache.mahout.matrix.Swapper swapper = new org.apache.mahout.matrix.Swapper() {
    public void swap(int a, int b) {
      long t2;  Object t1;
      t1 = v[a]; v[a] = v[b]; v[b] = t1;
      t2 = k[a]; k[a] = k[b];  k[b] = t2;
    }
  }; 

  org.apache.mahout.matrix.function.IntComparator comp = new org.apache.mahout.matrix.function.IntComparator() {
    public int compare(int a, int b) {
      int ab = ((Comparable)v[a]).compareTo((Comparable)v[b]);
      return ab<0 ? -1 : ab>0 ? 1 : (k[a]<k[b] ? -1 : (k[a]==k[b] ? 0 : 1));
      //return v[a]<v[b] ? -1 : v[a]>v[b] ? 1 : (k[a]<k[b] ? -1 : (k[a]==k[b] ? 0 : 1));
    }
  };

  org.apache.mahout.matrix.GenericSorting.quickSort(0,keyList.size(),comp,swapper);
}
/**
 * Associates the given key with the given value.
 * Replaces any old <tt>(key,someOtherValue)</tt> association, if existing.
 *
 * @param key the key the value shall be associated with.
 * @param value the value to be associated.
 * @return <tt>true</tt> if the receiver did not already contain such a key;
 *         <tt>false</tt> if the receiver did already contain such a key - the new value has now replaced the formerly associated value.
 */
public abstract boolean put(long key, Object value);
/**
 * Removes the given key with its associated element from the receiver, if present.
 *
 * @param key the key to be removed from the receiver.
 * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
 */
public abstract boolean removeKey(long key);
/**
 * Returns a string representation of the receiver, containing
 * the String representation of each key-value pair, sorted ascending by key.
 */
public String toString() {
  LongArrayList theKeys = keys();
  theKeys.sort();

  StringBuffer buf = new StringBuffer();
  buf.append("[");
  int maxIndex = theKeys.size() - 1;
  for (int i = 0; i <= maxIndex; i++) {
    long key = theKeys.get(i);
      buf.append(String.valueOf(key));
    buf.append("->");
      buf.append(String.valueOf(get(key)));
    if (i < maxIndex) buf.append(", ");
  }
  buf.append("]");
  return buf.toString();
}
/**
 * Returns a string representation of the receiver, containing
 * the String representation of each key-value pair, sorted ascending by value, according to natural ordering.
 */
public String toStringByValue() {
  LongArrayList theKeys = new LongArrayList();
  keysSortedByValue(theKeys);

  StringBuffer buf = new StringBuffer();
  buf.append("[");
  int maxIndex = theKeys.size() - 1;
  for (int i = 0; i <= maxIndex; i++) {
    long key = theKeys.get(i);
      buf.append(String.valueOf(key));
    buf.append("->");
      buf.append(String.valueOf(get(key)));
    if (i < maxIndex) buf.append(", ");
  }
  buf.append("]");
  return buf.toString();
}
/**
 * Returns a list filled with all values contained in the receiver.
 * The returned list has a size that equals <tt>this.size()</tt>.
 * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 * <p>
 * This method can be used to iterate over the values of the receiver.
 *
 * @return the values.
 */
public ObjectArrayList values() {
  ObjectArrayList list = new ObjectArrayList(size());
  values(list);
  return list;
}
/**
 * Fills all values contained in the receiver into the specified list.
 * Fills the list, starting at index 0.
 * After this call returns the specified list has a new size that equals <tt>this.size()</tt>.
 * Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link #forEachKey(LongProcedure)}.
 * <p>
 * This method can be used to iterate over the values of the receiver.
 *
 * @param list the list to be filled, can have any size.
 */
public void values(final ObjectArrayList list) {
  list.clear();
  forEachKey(
    new LongProcedure() {
      public boolean apply(long key) {
        list.add(get(key));
        return true;
      }
    }
  );
}
}
