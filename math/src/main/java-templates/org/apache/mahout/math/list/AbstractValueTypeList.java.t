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
package org.apache.mahout.math.list;
//CHECKSTYLE:OFF
import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.buffer.${valueTypeCap}BufferConsumer;
import org.apache.mahout.math.function.${valueTypeCap}Comparator;
import org.apache.mahout.math.function.${valueTypeCap}Procedure;
//CHECKSTYLE:ON

import java.util.ArrayList;
import java.util.List;

/**
 Abstract base class for resizable lists holding <code>${valueType}</code> elements; abstract.
*/
public abstract class Abstract${valueTypeCap}List extends AbstractList implements ${valueTypeCap}BufferConsumer, Cloneable {

  /**
   * The size of the list. This is a READ_ONLY variable for all methods but setSizeRaw(int newSize) !!! If you violate
   * this principle in subclasses, you should exactly know what you are doing.
   */
  protected int size;

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  public void add(${valueType} element) {
    beforeInsert(size, element);
  }

  /**
   * Appends all elements of the specified list to the receiver.
   *
   * @param other the list of which all elements shall be appended.
   */
  public void addAllOf(Abstract${valueTypeCap}List other) {
    addAllOfFromTo(other, 0, other.size() - 1);
  }

  /**
   * Appends the part of the specified list between <code>from</code> (inclusive) and <code>to</code> (inclusive) to the
   * receiver.
   *
   * @param other the list to be added to the receiver.
   * @param from  the index of the first element to be appended (inclusive).
   * @param to    the index of the last element to be appended (inclusive).
   * @throws IndexOutOfBoundsException index is out of range (<tt>other.size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=other.size())</tt>).
   */
  public void addAllOfFromTo(Abstract${valueTypeCap}List other, int from, int to) {
    beforeInsertAllOfFromTo(size, other, from, to);
  }
  
  /**
   * Appends the specified list to the end of this list.
   * @param other the list to be appended. 
   **/
  @Override
  public void addAllOf(${valueTypeCap}ArrayList other) {
    addAllOfFromTo(other, 0, other.size() - 1);
  }

  /**
   * Inserts the specified element before the specified position into the receiver. Shifts the element currently at that
   * position (if any) and any subsequent elements to the right.
   *
   * @param index   index before which the specified element is to be inserted (must be in [0,size]).
   * @param element element to be inserted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  public void beforeInsert(int index, ${valueType} element) {
    beforeInsertDummies(index, 1);
    set(index, element);
  }

  /**
   * Inserts the part of the specified list between <code>otherFrom</code> (inclusive) and <code>otherTo</code>
   * (inclusive) before the specified position into the receiver. Shifts the element currently at that position (if any)
   * and any subsequent elements to the right.
   *
   * @param index index before which to insert first element from the specified list (must be in [0,size])..
   * @param other list of which a part is to be inserted into the receiver.
   * @param from  the index of the first element to be inserted (inclusive).
   * @param to    the index of the last element to be inserted (inclusive).
   * @throws IndexOutOfBoundsException index is out of range (<tt>other.size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=other.size())</tt>).
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  public void beforeInsertAllOfFromTo(int index, Abstract${valueTypeCap}List other, int from, int to) {
    int length = to - from + 1;
    this.beforeInsertDummies(index, length);
    this.replaceFromToWithFrom(index, index + length - 1, other, from);
  }

  /**
   * Inserts <tt>length</tt> dummy elements before the specified position into the receiver. Shifts the element
   * currently at that position (if any) and any subsequent elements to the right. <b>This method must set the new size
   * to be <tt>size()+length</tt>.
   *
   * @param index  index before which to insert dummy elements (must be in [0,size])..
   * @param length number of dummy elements to be inserted.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt; size()</tt>.
   */
  @Override
  protected void beforeInsertDummies(int index, int length) {
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    if (length > 0) {
      ensureCapacity(size + length);
      setSizeRaw(size + length);
      replaceFromToWithFrom(index + length, size - 1, this, index);
    }
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm.  The receiver must
   * <strong>must</strong> be sorted (as by the sort method) prior to making this call.  If it is not sorted, the
   * results are undefined: in particular, the call may enter an infinite loop.  If the receiver contains multiple
   * elements equal to the specified object, there is no guarantee which instance will be found.
   *
   * @param key the value to be searched for.
   * @return index of the search key, if it is contained in the receiver; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>.  The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the receiver: the index of the first element greater than the key, or <tt>receiver.size()</tt>, if all
   *         elements in the receiver are less than the specified key.  Note that this guarantees that the return value
   *         will be &gt;= 0 if and only if the key is found.
   * @see java.util.Arrays
   */
  public int binarySearch(${valueType} key) {
    return this.binarySearchFromTo(key, 0, size - 1);
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm.  The receiver must
   * <strong>must</strong> be sorted (as by the sort method) prior to making this call.  If it is not sorted, the
   * results are undefined: in particular, the call may enter an infinite loop.  If the receiver contains multiple
   * elements equal to the specified object, there is no guarantee which instance will be found.
   *
   * @param key  the value to be searched for.
   * @param from the leftmost search position, inclusive.
   * @param to   the rightmost search position, inclusive.
   * @return index of the search key, if it is contained in the receiver; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>.  The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the receiver: the index of the first element greater than the key, or <tt>receiver.size()</tt>, if all
   *         elements in the receiver are less than the specified key.  Note that this guarantees that the return value
   *         will be &gt;= 0 if and only if the key is found.
   * @see java.util.Arrays
   */
  public int binarySearchFromTo(${valueType} key, int from, int to) {
    int low = from;
    int high = to;
    while (low <= high) {
      int mid = (low + high) / 2;
      ${valueType} midVal = get(mid);

      if (midVal < key) {
        low = mid + 1;
      } else if (midVal > key) {
        high = mid - 1;
      } else {
        return mid;
      } // key found
    }
    return -(low + 1);  // key not found.
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    return partFromTo(0, size - 1);
  }

  /**
   * Returns true if the receiver contains the specified element.
   *
   * @param elem element whose presence in the receiver is to be tested.
   */
  public boolean contains(${valueType} elem) {
    return indexOfFromTo(elem, 0, size - 1) >= 0;
  }

  /**
   * Deletes the first element from the receiver that is identical to the specified element. Does nothing, if no such
   * matching element is contained.
   *
   * @param element the element to be deleted.
   */
  public void delete(${valueType} element) {
    int index = indexOfFromTo(element, 0, size - 1);
    if (index >= 0) {
      remove(index);
    }
  }

  /**
   * Returns the elements currently stored, possibly including invalid elements between size and capacity.
   *
   * <b>WARNING:</b> For efficiency reasons and to keep memory usage low, this method may decide <b>not to copy the
   * array</b>. So if subsequently you modify the returned array directly via the [] operator, be sure you know what
   * you're doing.
   *
   * @return the elements currently stored.
   */
  public ${valueType}[] elements() {
    ${valueType}[] myElements = new ${valueType}[size];
    for (int i = size; --i >= 0;) {
      myElements[i] = getQuick(i);
    }
    return myElements;
  }

  /**
   * Sets the receiver's elements to be the specified array. The size and capacity of the list is the length of the
   * array. <b>WARNING:</b> For efficiency reasons and to keep memory usage low, this method may decide <b>not to copy
   * the array</b>. So if subsequently you modify the returned array directly via the [] operator, be sure you know what
   * you're doing.
   *
   * @param elements the new elements to be stored.
   * @return the receiver itself.
   */
  public Abstract${valueTypeCap}List elements(${valueType}[] elements) {
    clear();
    addAllOfFromTo(new ${valueTypeCap}ArrayList(elements), 0, elements.length - 1);
    return this;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  public abstract void ensureCapacity(int minCapacity);

  /**
   * Compares the specified Object with the receiver. Returns true if and only if the specified Object is also an
   * ArrayList of the same type, both Lists have the same size, and all corresponding pairs of elements in the two Lists
   * are identical. In other words, two Lists are defined to be equal if they contain the same elements in the same
   * order.
   *
   * @param otherObj the Object to be compared for equality with the receiver.
   * @return true if the specified Object is equal to the receiver.
   */
  public boolean equals(Object otherObj) { //delta
    if (otherObj == null) {
      return false;
    }
    if (!(otherObj instanceof Abstract${valueTypeCap}List)) {
      return false;
    }
    if (this == otherObj) {
      return true;
    }
    Abstract${valueTypeCap}List other = (Abstract${valueTypeCap}List) otherObj;
    if (size() != other.size()) {
      return false;
    }

    for (int i = size(); --i >= 0;) {
      if (getQuick(i) != other.getQuick(i)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Sets the specified range of elements in the specified array to the specified value.
   *
   * @param from the index of the first element (inclusive) to be filled with the specified value.
   * @param to   the index of the last element (inclusive) to be filled with the specified value.
   * @param val  the value to be stored in the specified elements of the receiver.
   */
  public void fillFromToWith(int from, int to, ${valueType} val) {
    checkRangeFromTo(from, to, this.size);
    for (int i = from; i <= to;) {
      setQuick(i++, val);
    }
  }

  /**
   * Applies a procedure to each element of the receiver, if any. Starts at index 0, moving rightwards.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  public boolean forEach(${valueTypeCap}Procedure procedure) {
    for (int i = 0; i < size;) {
      if (!procedure.apply(get(i++))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the element at the specified position in the receiver.
   *
   * @param index index of element to return.
   * @throws IndexOutOfBoundsException index is out of range (index &lt; 0 || index &gt;= size()).
   */
  public ${valueType} get(int index) {
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    return getQuick(index);
  }

  /**
   * Returns the element at the specified position in the receiver; <b>WARNING:</b> Does not check preconditions.
   * Provided with invalid parameters this method may return invalid elements without throwing any exception! <b>You
   * should only use this method when you are absolutely sure that the index is within bounds.</b> Precondition
   * (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * This method is normally only used internally in large loops where bounds are explicitly checked before the loop and
   * need no be rechecked within the loop. However, when desperately, you can give this method <tt>public</tt>
   * visibility in subclasses.
   *
   * @param index index of element to return.
   */
  protected abstract ${valueType} getQuick(int index);

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element.
   *
   * @param element the element to be searched for.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  public int indexOf(${valueType} element) { //delta
    return indexOfFromTo(element, 0, size - 1);
  }

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element. Searches between <code>from</code>, inclusive and <code>to</code>, inclusive. Tests for
   * identity.
   *
   * @param element element to search for.
   * @param from    the leftmost search position, inclusive.
   * @param to      the rightmost search position, inclusive.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public int indexOfFromTo(${valueType} element, int from, int to) {
    checkRangeFromTo(from, to, size);

    for (int i = from; i <= to; i++) {
      if (element == getQuick(i)) {
        return i;
      } //found
    }
    return -1; //not found
  }

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element.
   *
   * @param element the element to be searched for.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  public int lastIndexOf(${valueType} element) {
    return lastIndexOfFromTo(element, 0, size - 1);
  }

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element. Searches beginning at <code>to</code>, inclusive until <code>from</code>, inclusive. Tests
   * for identity.
   *
   * @param element element to search for.
   * @param from    the leftmost search position, inclusive.
   * @param to      the rightmost search position, inclusive.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public int lastIndexOfFromTo(${valueType} element, int from, int to) {
    checkRangeFromTo(from, to, size());

    for (int i = to; i >= from; i--) {
      if (element == getQuick(i)) {
        return i;
      } //found
    }
    return -1; //not found
  }

  /**
   * Sorts the specified range of the receiver into ascending order.
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  @Override
  public void mergeSortFromTo(int from, int to) {
    int mySize = size();
    checkRangeFromTo(from, to, mySize);

    ${valueType}[] myElements = elements();
    Sorting.mergeSort(myElements, from, to + 1);
    elements(myElements);
    setSizeRaw(mySize);
  }

  /**
   * Sorts the receiver according to the order induced by the specified comparator.  All elements in the range must be
   * <i>mutually comparable</i> by the specified comparator (that is, <tt>c.compare(e1, e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the range).<p>
   *
   * This sort is guaranteed to be <i>stable</i>:  equal elements will not be reordered as a result of the sort.<p>
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @param c    the comparator to determine the order of the receiver.
   * @throws ClassCastException             if the array contains elements that are not <i>mutually comparable</i> using
   *                                        the specified comparator.
   * @throws IllegalArgumentException       if <tt>fromIndex &gt; toIndex</tt>
   * @throws ArrayIndexOutOfBoundsException if <tt>fromIndex &lt; 0</tt> or <tt>toIndex &gt; a.length</tt>
   * @throws IndexOutOfBoundsException      index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                        to&gt;=size())</tt>).
   */
  public void mergeSortFromTo(int from, int to, ${valueTypeCap}Comparator c) {
    int mySize = size();
    checkRangeFromTo(from, to, mySize);

    ${valueType}[] myElements = elements();
    Sorting.mergeSort(myElements, from, to + 1, c);
    elements(myElements);
    setSizeRaw(mySize);
  }

  /**
   * Returns a new list of the part of the receiver between <code>from</code>, inclusive, and <code>to</code>,
   * inclusive.
   *
   * @param from the index of the first element (inclusive).
   * @param to   the index of the last element (inclusive).
   * @return a new list
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public Abstract${valueTypeCap}List partFromTo(int from, int to) {
    checkRangeFromTo(from, to, size);

    int length = to - from + 1;
    ${valueTypeCap}ArrayList part = new ${valueTypeCap}ArrayList(length);
    part.addAllOfFromTo(this, from, to);
    return part;
  }
  
  /**
   * Sorts the specified range of the receiver into ascending numerical order.  The sorting algorithm is a tuned
   * quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's "Engineering a Sort Function", Software-Practice
   * and Experience, Vol. 23(11) P. 1249-1265 (November 1993).  This algorithm offers n*log(n) performance on many data
   * sets that cause other quicksorts to degrade to quadratic performance.
   *
   * <p><b>You should never call this method unless you are sure that this particular sorting algorithm is the right one
   * for your data set.</b> It is generally better to call <tt>sort()</tt> or <tt>sortFromTo(...)</tt> instead, because
   * those methods automatically choose the best sorting algorithm.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  @Override
  public void quickSortFromTo(int from, int to) {
    int mySize = size();
    checkRangeFromTo(from, to, mySize);

    ${valueType}[] myElements = elements();
    java.util.Arrays.sort(myElements, from, to + 1);
    elements(myElements);
    setSizeRaw(mySize);
  }

  /**
   * Sorts the receiver according to the order induced by the specified comparator.  All elements in the range must be
   * <i>mutually comparable</i> by the specified comparator (that is, <tt>c.compare(e1, e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the range).<p>
   *
   * The sorting algorithm is a tuned quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's "Engineering a
   * Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265 (November 1993).  This algorithm offers
   * n*log(n) performance on many data sets that cause other quicksorts to degrade to quadratic performance.
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @param c    the comparator to determine the order of the receiver.
   * @throws ClassCastException             if the array contains elements that are not <i>mutually comparable</i> using
   *                                        the specified comparator.
   * @throws IllegalArgumentException       if <tt>fromIndex &gt; toIndex</tt>
   * @throws ArrayIndexOutOfBoundsException if <tt>fromIndex &lt; 0</tt> or <tt>toIndex &gt; a.length</tt>
   * @throws IndexOutOfBoundsException      index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                        to&gt;=size())</tt>).
   */
  public void quickSortFromTo(int from, int to, ${valueTypeCap}Comparator c) {
    int mySize = size();
    checkRangeFromTo(from, to, mySize);

    ${valueType}[] myElements = elements();
    Sorting.quickSort(myElements, from, to + 1, c);
    elements(myElements);
    setSizeRaw(mySize);
  }

  /**
   * Removes from the receiver all elements that are contained in the specified list. Tests for identity.
   *
   * @param other the other list.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  public boolean removeAll(Abstract${valueTypeCap}List other) {
    if (other.isEmpty()) {
      return false;
    } //nothing to do
    int limit = other.size() - 1;
    int j = 0;

    for (int i = 0; i < size; i++) {
      if (other.indexOfFromTo(getQuick(i), 0, limit) < 0) {
        setQuick(j++, getQuick(i));
      }
    }

    boolean modified = (j != size);
    setSize(j);
    return modified;
  }

  /**
   * Removes from the receiver all elements whose index is between <code>from</code>, inclusive and <code>to</code>,
   * inclusive.  Shifts any succeeding elements to the left (reduces their index). This call shortens the list by
   * <tt>(to - from + 1)</tt> elements.
   *
   * @param from index of first element to be removed.
   * @param to   index of last element to be removed.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  @Override
  public void removeFromTo(int from, int to) {
    checkRangeFromTo(from, to, size);
    int numMoved = size - to - 1;
    if (numMoved > 0) {
      replaceFromToWithFrom(from, from - 1 + numMoved, this, to + 1);
      //fillFromToWith(from+numMoved, size-1, 0.0f); //delta
    }
    int width = to - from + 1;
    if (width > 0) {
      setSizeRaw(size - width);
    }
  }

  /**
   * Replaces a number of elements in the receiver with the same number of elements of another list. Replaces elements
   * in the receiver, between <code>from</code> (inclusive) and <code>to</code> (inclusive), with elements of
   * <code>other</code>, starting from <code>otherFrom</code> (inclusive).
   *
   * @param from      the position of the first element to be replaced in the receiver
   * @param to        the position of the last element to be replaced in the receiver
   * @param other     list holding elements to be copied into the receiver.
   * @param otherFrom position of first element within other list to be copied.
   */
  public void replaceFromToWithFrom(int from, int to, Abstract${valueTypeCap}List other, int otherFrom) {
    int length = to - from + 1;
    if (length > 0) {
      checkRangeFromTo(from, to, size());
      checkRangeFromTo(otherFrom, otherFrom + length - 1, other.size());

      // unambiguous copy (it may hold other==this)
      if (from <= otherFrom) {
        while (--length >= 0) {
          setQuick(from++, other.getQuick(otherFrom++));
        }
      } else {
        int otherTo = otherFrom + length - 1;
        while (--length >= 0) {
          setQuick(to--, other.getQuick(otherTo--));
        }
      }
    }
  }

  /**
   * Replaces the part between <code>from</code> (inclusive) and <code>to</code> (inclusive) with the other list's part
   * between <code>otherFrom</code> and <code>otherTo</code>. Powerful (and tricky) method! Both parts need not be of
   * the same size (part A can both be smaller or larger than part B). Parts may overlap. Receiver and other list may
   * (but most not) be identical. If <code>from &gt; to</code>, then inserts other part before <code>from</code>.
   *
   * @param from      the first element of the receiver (inclusive)
   * @param to        the last element of the receiver (inclusive)
   * @param other     the other list (may be identical with receiver)
   * @param otherFrom the first element of the other list (inclusive)
   * @param otherTo   the last element of the other list (inclusive)
   *
   *                  <p><b>Examples:</b><pre>
   *                                                                     a=[0, 1, 2, 3, 4, 5, 6, 7]
   *                                                                     b=[50, 60, 70, 80, 90]
   *                                                                     a.R(...)=a.replaceFromToWithFromTo(...)
   *
   *                                                                     a.R(3,5,b,0,4)-->[0, 1, 2, 50, 60, 70, 80, 90,
   *                  6, 7]
   *                                                                     a.R(1,6,b,0,4)-->[0, 50, 60, 70, 80, 90, 7]
   *                                                                     a.R(0,6,b,0,4)-->[50, 60, 70, 80, 90, 7]
   *                                                                     a.R(3,5,b,1,2)-->[0, 1, 2, 60, 70, 6, 7]
   *                                                                     a.R(1,6,b,1,2)-->[0, 60, 70, 7]
   *                                                                     a.R(0,6,b,1,2)-->[60, 70, 7]
   *                                                                     a.R(5,3,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70,
   *                  80, 90, 5, 6, 7]
   *                                                                     a.R(5,0,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70,
   *                  80, 90, 5, 6, 7]
   *                                                                     a.R(5,3,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6,
   *                  7]
   *                                                                     a.R(5,0,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6,
   *                  7]
   *
   *                                                                     Extreme cases:
   *                                                                     a.R(5,3,b,0,0)-->[0, 1, 2, 3, 4, 50, 5, 6, 7]
   *                                                                     a.R(5,3,b,4,4)-->[0, 1, 2, 3, 4, 90, 5, 6, 7]
   *                                                                     a.R(3,5,a,0,1)-->[0, 1, 2, 0, 1, 6, 7]
   *                                                                     a.R(3,5,a,3,5)-->[0, 1, 2, 3, 4, 5, 6, 7]
   *                                                                     a.R(3,5,a,4,4)-->[0, 1, 2, 4, 6, 7]
   *                                                                     a.R(5,3,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
   *                  5, 6, 7]
   *                                                                     a.R(0,-1,b,0,4)-->[50, 60, 70, 80, 90, 0, 1, 2,
   *                  3, 4, 5, 6, 7]
   *                                                                     a.R(0,-1,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3,
   *                  4, 5, 6, 7]
   *                                                                     a.R(8,0,a,0,4)-->[0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
   *                  2, 3, 4]
   *                                                                     </pre>
   */
  public void replaceFromToWithFromTo(int from, int to, Abstract${valueTypeCap}List other, int otherFrom, int otherTo) {
    if (otherFrom > otherTo) {
      throw new IndexOutOfBoundsException("otherFrom: " + otherFrom + ", otherTo: " + otherTo);
    }

    if (this == other && to - from != otherTo - otherFrom) { // avoid stumbling over my own feet
      replaceFromToWithFromTo(from, to, partFromTo(otherFrom, otherTo), 0, otherTo - otherFrom);
      return;
    }

    int length = otherTo - otherFrom + 1;
    int diff = length;
    int theLast = from - 1;

    if (to >= from) {
      diff -= (to - from + 1);
      theLast = to;
    }

    if (diff > 0) {
      beforeInsertDummies(theLast + 1, diff);
    } else {
      if (diff < 0) {
        removeFromTo(theLast + diff, theLast - 1);
      }
    }

    if (length > 0) {
      replaceFromToWithFrom(from, from + length - 1, other, otherFrom);
    }
  }
  
  /**
   * Retains (keeps) only the elements in the receiver that are contained in the specified other list. In other words,
   * removes from the receiver all of its elements that are not contained in the specified other list.
   *
   * @param other the other list to test against.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  public boolean retainAll(Abstract${valueTypeCap}List other) {
    if (other.isEmpty()) {
      if (size == 0) {
        return false;
      }
      setSize(0);
      return true;
    }

    int limit = other.size() - 1;
    int j = 0;
    for (int i = 0; i < size; i++) {
      if (other.indexOfFromTo(getQuick(i), 0, limit) >= 0) {
        setQuick(j++, getQuick(i));
      }
    }

    boolean modified = (j != size);
    setSize(j);
    return modified;
  }
  
  /** Reverses the elements of the receiver. Last becomes first, second last becomes second first, and so on. */
  @Override
  public void reverse() {
    int limit = size() / 2;
    int j = size() - 1;

    for (int i = 0; i < limit;) { //swap
      ${valueType} tmp = getQuick(i);
      setQuick(i++, getQuick(j));
      setQuick(j--, tmp);
    }
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   * @throws IndexOutOfBoundsException if <tt>index &lt; 0 || index &gt;= size()</tt>.
   */
  public void set(int index, ${valueType} element) {
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    setQuick(index, element);
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element; <b>WARNING:</b> Does not
   * check preconditions. Provided with invalid parameters this method may access invalid indexes without throwing any
   * exception! <b>You should only use this method when you are absolutely sure that the index is within bounds.</b>
   * Precondition (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * This method is normally only used internally in large loops where bounds are explicitly checked before the loop and
   * need no be rechecked within the loop. However, when desperately, you can give this method <tt>public</tt>
   * visibility in subclasses.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   */
  protected abstract void setQuick(int index, ${valueType} element);

  /**
   * Sets the size of the receiver without modifying it otherwise. This method should not release or allocate new memory
   * but simply set some instance variable like <tt>size</tt>.
   *
   * If your subclass overrides and delegates size changing methods to some other object, you must make sure that those
   * overriding methods not only update the size of the delegate but also of this class. For example: public
   * DatabaseList extends Abstract${valueTypeCap}List { ... public void removeFromTo(int from,int to) {
   * myDatabase.removeFromTo(from,to); this.setSizeRaw(size-(to-from+1)); } }
   */
  protected void setSizeRaw(int newSize) {
    size = newSize;
  }

  /** Returns the number of elements contained in the receiver. */
  @Override
  public int size() {
    return size;
  }

  /**
   * Returns a list which is a concatenation of <code>times</code> times the receiver.
   *
   * @param times the number of times the receiver shall be copied.
   */
  public Abstract${valueTypeCap}List times(int times) {
    Abstract${valueTypeCap}List newList = new ${valueTypeCap}ArrayList(times * size());
    for (int i = times; --i >= 0;) {
      newList.addAllOfFromTo(this, 0, size() - 1);
    }
    return newList;
  }

  /** Returns a <code>ArrayList</code> containing all the elements in the receiver. */
  public List<${valueObjectType}> toList() {
    int mySize = size();
    List<${valueObjectType}> list = new ArrayList<${valueObjectType}>(mySize);
    for (int i = 0; i < mySize; i++) {
      list.add(get(i));
    }
    return list;
  }
  
  public ${valueType}[] toArray(${valueType}[] values) {
   int mySize = size();
   ${valueType}[] myElements;
   if (values.length >= mySize) {
     myElements = values;
   } else {
     myElements = new ${valueType}[mySize];
   }
   for (int i = size; --i >= 0;) {
      myElements[i] = getQuick(i);
    }
    return myElements;
  }

  /** Returns a string representation of the receiver, containing the String representation of each element. */
  public String toString() {
    return org.apache.mahout.math.Arrays.toString(partFromTo(0, size() - 1).elements());
  }
}
