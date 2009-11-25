/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.list;

import org.apache.mahout.matrix.function.ObjectProcedure;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
/**
 Resizable list holding <code>Object</code> elements; implemented with arrays.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class ObjectArrayList extends AbstractList<Object[]> {

  /**
   * The array buffer into which the elements of the list are stored. The capacity of the list is the length of this
   * array buffer.
   */
  protected Object[] elements;

  /** The size of the list. */
  protected int size;

  /** Constructs an empty list. */
  public ObjectArrayList() {
    this(10);
  }

  /**
   * Constructs a list containing the specified elements. The initial size and capacity of the list is the length of the
   * array.
   *
   * <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the array is not copied</b>. So if
   * subsequently you modify the specified array directly via the [] operator, be sure you know what you're doing.
   *
   * @param elements the array to be backed by the the constructed list
   */
  public ObjectArrayList(Object[] elements) {
    elements(elements);
  }

  /**
   * Constructs an empty list with the specified initial capacity.
   *
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  public ObjectArrayList(int initialCapacity) {
    this(new Object[initialCapacity]);
    size = 0;
  }

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  public void add(Object element) {
    if (size == elements.length) {
      ensureCapacity(size + 1);
    }
    elements[size++] = element;
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
  public void addAllOfFromTo(ObjectArrayList other, int from, int to) {
    beforeInsertAllOfFromTo(size, other, from, to);
  }

  /**
   * Inserts the specified element before the specified position into the receiver. Shifts the element currently at that
   * position (if any) and any subsequent elements to the right.
   *
   * @param index   index before which the specified element is to be inserted (must be in [0,size]).
   * @param element element to be inserted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  public void beforeInsert(int index, Object element) {
    // overridden for performance only.
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    ensureCapacity(size + 1);
    System.arraycopy(elements, index, elements, index + 1, size - index);
    elements[index] = element;
    size++;
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
  public void beforeInsertAllOfFromTo(int index, ObjectArrayList other, int from, int to) {
    int length = to - from + 1;
    this.beforeInsertDummies(index, length);
    this.replaceFromToWithFrom(index, index + length - 1, other, from);
  }

  /**
   * Inserts length dummies before the specified position into the receiver. Shifts the element currently at that
   * position (if any) and any subsequent elements to the right.
   *
   * @param index  index before which to insert dummies (must be in [0,size])..
   * @param length number of dummies to be inserted.
   */
  @Override
  protected void beforeInsertDummies(int index, int length) {
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    if (length > 0) {
      ensureCapacity(size + length);
      System.arraycopy(elements, index, elements, index + length, size - index);
      size += length;
    }
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm. The receiver must be sorted into
   * ascending order according to the <i>natural ordering</i> of its elements (as by the sort method) prior to making
   * this call. If it is not sorted, the results are undefined: in particular, the call may enter an infinite loop.  If
   * the receiver contains multiple elements equal to the specified object, there is no guarantee which instance will be
   * found.
   *
   * @param key the value to be searched for.
   * @return index of the search key, if it is contained in the receiver; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>.  The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the receiver: the index of the first element greater than the key, or <tt>receiver.size()</tt>, if all
   *         elements in the receiver are less than the specified key.  Note that this guarantees that the return value
   *         will be &gt;= 0 if and only if the key is found.
   * @see Comparable
   * @see java.util.Arrays
   */
  public int binarySearch(Object key) {
    return this.binarySearchFromTo(key, 0, size - 1);
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm. The receiver must be sorted into
   * ascending order according to the <i>natural ordering</i> of its elements (as by the sort method) prior to making
   * this call. If it is not sorted, the results are undefined: in particular, the call may enter an infinite loop.  If
   * the receiver contains multiple elements equal to the specified object, there is no guarantee which instance will be
   * found.
   *
   * @param key  the value to be searched for.
   * @param from the leftmost search position, inclusive.
   * @param to   the rightmost search position, inclusive.
   * @return index of the search key, if it is contained in the receiver; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>.  The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the receiver: the index of the first element greater than the key, or <tt>receiver.size()</tt>, if all
   *         elements in the receiver are less than the specified key.  Note that this guarantees that the return value
   *         will be &gt;= 0 if and only if the key is found.
   * @see Comparable
   * @see java.util.Arrays
   */
  public int binarySearchFromTo(Object key, int from, int to) {
    int low = from;
    int high = to;

    while (low <= high) {
      int mid = (low + high) / 2;
      Object midVal = elements[mid];
      int cmp = ((Comparable<Object>) midVal).compareTo(key);

      if (cmp < 0) {
        low = mid + 1;
      } else if (cmp > 0) {
        high = mid - 1;
      } else {
        return mid;
      } // key found
    }
    return -(low + 1);  // key not found.
  }

  /**
   * Searches the receiver for the specified value using the binary search algorithm. The receiver must be sorted into
   * ascending order according to the specified comparator.  All elements in the range must be <i>mutually
   * comparable</i> by the specified comparator (that is, <tt>c.compare(e1, e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the range).<p>
   *
   * If the receiver is not sorted, the results are undefined: in particular, the call may enter an infinite loop.  If
   * the receiver contains multiple elements equal to the specified object, there is no guarantee which instance will be
   * found.
   *
   * @param key        the value to be searched for.
   * @param from       the leftmost search position, inclusive.
   * @param to         the rightmost search position, inclusive.
   * @param comparator the comparator by which the receiver is sorted.
   * @return index of the search key, if it is contained in the receiver; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>.  The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the receiver: the index of the first element greater than the key, or <tt>receiver.size()</tt>, if all
   *         elements in the receiver are less than the specified key.  Note that this guarantees that the return value
   *         will be &gt;= 0 if and only if the key is found.
   * @throws ClassCastException if the receiver contains elements that are not <i>mutually comparable</i> using the
   *                            specified comparator.
   * @see org.apache.mahout.matrix.Sorting
   * @see java.util.Arrays
   * @see java.util.Comparator
   */
  public int binarySearchFromTo(Object key, int from, int to, Comparator<Object> comparator) {
    return org.apache.mahout.matrix.Sorting.binarySearchFromTo(this.elements, key, from, to, comparator);
  }

  /**
   * Returns a copy of the receiver such that the copy and the receiver <i>share</i> the same elements, but do not share
   * the same array to index them; So modifying an object in the copy modifies the object in the receiver and vice
   * versa; However, structurally modifying the copy (for example changing its size, setting other objects at indexes,
   * etc.) does not affect the receiver and vice versa.
   *
   * @return a copy of the receiver.
   */
  @Override
  public Object clone() {
    ObjectArrayList v = (ObjectArrayList) super.clone();
    v.elements = elements.clone();
    return v;
  }

  /**
   * Returns true if the receiver contains the specified element. Tests for equality or identity as specified by
   * testForEquality.
   *
   * @param elem            element to search for.
   * @param testForEquality if true -> test for equality, otherwise for identity.
   */
  public boolean contains(Object elem, boolean testForEquality) {
    return indexOfFromTo(elem, 0, size - 1, testForEquality) >= 0;
  }

  /**
   * Returns a copy of the receiver; call <code>clone()</code> and casts the result. Returns a copy such that the copy
   * and the receiver <i>share</i> the same elements, but do not share the same array to index them; So modifying an
   * object in the copy modifies the object in the receiver and vice versa; However, structurally modifying the copy
   * (for example changing its size, setting other objects at indexes, etc.) does not affect the receiver and vice
   * versa.
   *
   * @return a copy of the receiver.
   */
  public ObjectArrayList copy() {
    return (ObjectArrayList) clone();
  }

  /**
   * Deletes the first element from the receiver that matches the specified element. Does nothing, if no such matching
   * element is contained.
   *
   * Tests elements for equality or identity as specified by <tt>testForEquality</tt>. When testing for equality, two
   * elements <tt>e1</tt> and <tt>e2</tt> are <i>equal</i> if <tt>(e1==null ? e2==null : e1.equals(e2))</tt>.)
   *
   * @param testForEquality if true -> tests for equality, otherwise for identity.
   * @param element         the element to be deleted.
   */
  public void delete(Object element, boolean testForEquality) {
    int index = indexOfFromTo(element, 0, size - 1, testForEquality);
    if (index >= 0) {
      removeFromTo(index, index);
    }
  }

  /**
   * Returns the elements currently stored, including invalid elements between size and capacity, if any.
   *
   * <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the array is not copied</b>. So if
   * subsequently you modify the returned array directly via the [] operator, be sure you know what you're doing.
   *
   * @return the elements currently stored.
   */
  public Object[] elements() {
    return elements;
  }

  /**
   * Sets the receiver's elements to be the specified array (not a copy of it).
   *
   * The size and capacity of the list is the length of the array. <b>WARNING:</b> For efficiency reasons and to keep
   * memory usage low, <b>the array is not copied</b>. So if subsequently you modify the specified array directly via
   * the [] operator, be sure you know what you're doing.
   *
   * @param elements the new elements to be stored.
   * @return the receiver itself.
   */
  public ObjectArrayList elements(Object[] elements) {
    this.elements = elements;
    this.size = elements.length;
    return this;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  public void ensureCapacity(int minCapacity) {
    elements = org.apache.mahout.matrix.Arrays.ensureCapacity(elements, minCapacity);
  }

  /**
   * Compares the specified Object with the receiver for equality. Returns true if and only if the specified Object is
   * also an ObjectArrayList, both lists have the same size, and all corresponding pairs of elements in the two lists
   * are equal. In other words, two lists are defined to be equal if they contain the same elements in the same order.
   * Two elements <tt>e1</tt> and <tt>e2</tt> are <i>equal</i> if <tt>(e1==null ? e2==null : e1.equals(e2))</tt>.)
   *
   * @param otherObj the Object to be compared for equality with the receiver.
   * @return true if the specified Object is equal to the receiver.
   */
  public boolean equals(Object otherObj) { //delta
    return equals(otherObj, true);
  }

  /**
   * Compares the specified Object with the receiver for equality. Returns true if and only if the specified Object is
   * also an ObjectArrayList, both lists have the same size, and all corresponding pairs of elements in the two lists
   * are the same. In other words, two lists are defined to be equal if they contain the same elements in the same
   * order. Tests elements for equality or identity as specified by <tt>testForEquality</tt>. When testing for equality,
   * two elements <tt>e1</tt> and <tt>e2</tt> are <i>equal</i> if <tt>(e1==null ? e2==null : e1.equals(e2))</tt>.)
   *
   * @param otherObj        the Object to be compared for equality with the receiver.
   * @param testForEquality if true -> tests for equality, otherwise for identity.
   * @return true if the specified Object is equal to the receiver.
   */
  public boolean equals(Object otherObj, boolean testForEquality) { //delta
    if (!(otherObj instanceof ObjectArrayList)) {
      return false;
    }
    if (this == otherObj) {
      return true;
    }
    if (otherObj == null) {
      return false;
    }
    ObjectArrayList other = (ObjectArrayList) otherObj;
    if (elements == other.elements()) {
      return true;
    }
    if (size != other.size()) {
      return false;
    }

    Object[] otherElements = other.elements();
    Object[] theElements = elements;
    if (!testForEquality) {
      for (int i = size; --i >= 0;) {
        if (theElements[i] != otherElements[i]) {
          return false;
        }
      }
    } else {
      for (int i = size; --i >= 0;) {
        if (!(theElements[i] == null ? otherElements[i] == null : theElements[i].equals(otherElements[i]))) {
          return false;
        }
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
  public void fillFromToWith(int from, int to, Object val) {
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
  public boolean forEach(ObjectProcedure procedure) {
    Object[] theElements = elements;
    int theSize = size;

    for (int i = 0; i < theSize;) {
      if (!procedure.apply(theElements[i++])) {
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
  public Object get(int index) {
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    return elements[index];
  }

  /**
   * Returns the element at the specified position in the receiver; <b>WARNING:</b> Does not check preconditions.
   * Provided with invalid parameters this method may return invalid elements without throwing any exception! <b>You
   * should only use this method when you are absolutely sure that the index is within bounds.</b> Precondition
   * (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * @param index index of element to return.
   */
  public Object getQuick(int index) {
    return elements[index];
  }

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element.
   *
   * Tests for equality or identity as specified by testForEquality.
   *
   * @param testForEquality if <code>true</code> -> test for equality, otherwise for identity.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  public int indexOf(Object element, boolean testForEquality) {
    return this.indexOfFromTo(element, 0, size - 1, testForEquality);
  }

  /**
   * Returns the index of the first occurrence of the specified element. Returns <code>-1</code> if the receiver does
   * not contain this element. Searches between <code>from</code>, inclusive and <code>to</code>, inclusive.
   *
   * Tests for equality or identity as specified by <code>testForEquality</code>.
   *
   * @param element         element to search for.
   * @param from            the leftmost search position, inclusive.
   * @param to              the rightmost search position, inclusive.
   * @param testForEquality if </code>true</code> -> test for equality, otherwise for identity.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public int indexOfFromTo(Object element, int from, int to, boolean testForEquality) {
    if (size == 0) {
      return -1;
    }
    checkRangeFromTo(from, to, size);

    Object[] theElements = elements;
    if (testForEquality && element != null) {
      for (int i = from; i <= to; i++) {
        if (element.equals(theElements[i])) {
          return i;
        } //found
      }

    } else {
      for (int i = from; i <= to; i++) {
        if (element == theElements[i]) {
          return i;
        } //found
      }
    }
    return -1; //not found
  }

  /**
   * Determines whether the receiver is sorted ascending, according to the <i>natural ordering</i> of its elements.  All
   * elements in this range must implement the <tt>Comparable</tt> interface.  Furthermore, all elements in this range
   * must be <i>mutually comparable</i> (that is, <tt>e1.compareTo(e2)</tt> must not throw a <tt>ClassCastException</tt>
   * for any elements <tt>e1</tt> and <tt>e2</tt> in the array).<p>
   *
   * @param from the index of the first element (inclusive) to be sorted.
   * @param to   the index of the last element (inclusive) to be sorted.
   * @return <tt>true</tt> if the receiver is sorted ascending, <tt>false</tt> otherwise.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public boolean isSortedFromTo(int from, int to) {
    if (size == 0) {
      return true;
    }
    checkRangeFromTo(from, to, size);

    Object[] theElements = elements;
    for (int i = from + 1; i <= to; i++) {
      if (((Comparable<Object>) theElements[i]).compareTo(theElements[i - 1]) < 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element. Tests for equality or identity as specified by <code>testForEquality</code>.
   *
   * @param element         the element to be searched for.
   * @param testForEquality if <code>true</code> -> test for equality, otherwise for identity.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   */
  public int lastIndexOf(Object element, boolean testForEquality) {
    return lastIndexOfFromTo(element, 0, size - 1, testForEquality);
  }

  /**
   * Returns the index of the last occurrence of the specified element. Returns <code>-1</code> if the receiver does not
   * contain this element. Searches beginning at <code>to</code>, inclusive until <code>from</code>, inclusive. Tests
   * for equality or identity as specified by <code>testForEquality</code>.
   *
   * @param element         element to search for.
   * @param from            the leftmost search position, inclusive.
   * @param to              the rightmost search position, inclusive.
   * @param testForEquality if <code>true</code> -> test for equality, otherwise for identity.
   * @return the index of the last occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  public int lastIndexOfFromTo(Object element, int from, int to, boolean testForEquality) {
    if (size == 0) {
      return -1;
    }
    checkRangeFromTo(from, to, size);

    Object[] theElements = elements;
    if (testForEquality && element != null) {
      for (int i = to; i >= from; i--) {
        if (element.equals(theElements[i])) {
          return i;
        } //found
      }

    } else {
      for (int i = to; i >= from; i--) {
        if (element == theElements[i]) {
          return i;
        } //found
      }
    }
    return -1; //not found
  }

  /**
   * Sorts the specified range of the receiver into ascending order, according to the <i>natural ordering</i> of its
   * elements.  All elements in this range must implement the <tt>Comparable</tt> interface.  Furthermore, all elements
   * in this range must be <i>mutually comparable</i> (that is, <tt>e1.compareTo(e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the array).<p>
   *
   * This sort is guaranteed to be <i>stable</i>:  equal elements will not be reordered as a result of the sort.<p>
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
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);
    java.util.Arrays.sort(elements, from, to + 1);
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
   * @see Comparator
   */
  public void mergeSortFromTo(int from, int to, Comparator<Object> c) {
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);
    Arrays.sort(elements, from, to + 1, c);
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
  public ObjectArrayList partFromTo(int from, int to) {
    if (size == 0) {
      return new ObjectArrayList(0);
    }

    checkRangeFromTo(from, to, size);

    Object[] part = new Object[to - from + 1];
    System.arraycopy(elements, from, part, 0, to - from + 1);
    return new ObjectArrayList(part);
  }

  /**
   * Sorts the specified range of the receiver into ascending order, according to the <i>natural ordering</i> of its
   * elements.  All elements in this range must implement the <tt>Comparable</tt> interface.  Furthermore, all elements
   * in this range must be <i>mutually comparable</i> (that is, <tt>e1.compareTo(e2)</tt> must not throw a
   * <tt>ClassCastException</tt> for any elements <tt>e1</tt> and <tt>e2</tt> in the array).<p>
   *
   * The sorting algorithm is a tuned quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's "Engineering a
   * Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265 (November 1993).  This algorithm offers
   * n*log(n) performance on many data sets that cause other quicksorts to degrade to quadratic performance.
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
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);
    org.apache.mahout.matrix.Sorting.quickSort(elements, from, to + 1);
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
   * @see Comparator
   */
  public void quickSortFromTo(int from, int to, Comparator<Object> c) {
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);
    org.apache.mahout.matrix.Sorting.quickSort(elements, from, to + 1, c);
  }

  /**
   * Removes from the receiver all elements that are contained in the specified list. Tests for equality or identity as
   * specified by <code>testForEquality</code>.
   *
   * @param other           the other list.
   * @param testForEquality if <code>true</code> -> test for equality, otherwise for identity.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  public boolean removeAll(ObjectArrayList other, boolean testForEquality) {
    if (other.size == 0) {
      return false;
    } //nothing to do
    int limit = other.size - 1;
    int j = 0;
    Object[] theElements = elements;
    for (int i = 0; i < size; i++) {
      if (other.indexOfFromTo(theElements[i], 0, limit, testForEquality) < 0) {
        theElements[j++] = theElements[i];
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
    if (numMoved >= 0) {
      System.arraycopy(elements, to + 1, elements, from, numMoved);
      fillFromToWith(from + numMoved, size - 1, null); //delta
    }
    int width = to - from + 1;
    if (width > 0) {
      size -= width;
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
  public void replaceFromToWithFrom(int from, int to, ObjectArrayList other, int otherFrom) {
    int length = to - from + 1;
    if (length > 0) {
      checkRangeFromTo(from, to, size);
      checkRangeFromTo(otherFrom, otherFrom + length - 1, other.size);
      System.arraycopy(other.elements, otherFrom, elements, from, length);
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
   *                                                    a=[0, 1, 2, 3, 4, 5, 6, 7]
   *                                                    b=[50, 60, 70, 80, 90]
   *                                                    a.R(...)=a.replaceFromToWithFromTo(...)
   *
   *                                                    a.R(3,5,b,0,4)-->[0, 1, 2, 50, 60, 70, 80, 90, 6, 7]
   *                                                    a.R(1,6,b,0,4)-->[0, 50, 60, 70, 80, 90, 7]
   *                                                    a.R(0,6,b,0,4)-->[50, 60, 70, 80, 90, 7]
   *                                                    a.R(3,5,b,1,2)-->[0, 1, 2, 60, 70, 6, 7]
   *                                                    a.R(1,6,b,1,2)-->[0, 60, 70, 7]
   *                                                    a.R(0,6,b,1,2)-->[60, 70, 7]
   *                                                    a.R(5,3,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70, 80, 90, 5, 6, 7]
   *                                                    a.R(5,0,b,0,4)-->[0, 1, 2, 3, 4, 50, 60, 70, 80, 90, 5, 6, 7]
   *                                                    a.R(5,3,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6, 7]
   *                                                    a.R(5,0,b,1,2)-->[0, 1, 2, 3, 4, 60, 70, 5, 6, 7]
   *
   *                                                    Extreme cases:
   *                                                    a.R(5,3,b,0,0)-->[0, 1, 2, 3, 4, 50, 5, 6, 7]
   *                                                    a.R(5,3,b,4,4)-->[0, 1, 2, 3, 4, 90, 5, 6, 7]
   *                                                    a.R(3,5,a,0,1)-->[0, 1, 2, 0, 1, 6, 7]
   *                                                    a.R(3,5,a,3,5)-->[0, 1, 2, 3, 4, 5, 6, 7]
   *                                                    a.R(3,5,a,4,4)-->[0, 1, 2, 4, 6, 7]
   *                                                    a.R(5,3,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7]
   *                                                    a.R(0,-1,b,0,4)-->[50, 60, 70, 80, 90, 0, 1, 2, 3, 4, 5, 6, 7]
   *                                                    a.R(0,-1,a,0,4)-->[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7]
   *                                                    a.R(8,0,a,0,4)-->[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]
   *                                                    </pre>
   */
  public void replaceFromToWithFromTo(int from, int to, ObjectArrayList other, int otherFrom, int otherTo) {
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

    //System.out.println("from="+from);
    //System.out.println("to="+to);
    //System.out.println("diff="+diff);

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
      System.arraycopy(other.elements, otherFrom, elements, from, length);
    }
  }

  /**
   * Replaces the part of the receiver starting at <code>from</code> (inclusive) with all the elements of the specified
   * collection. Does not alter the size of the receiver. Replaces exactly <tt>Math.max(0,Math.min(size()-from,
   * other.size()))</tt> elements.
   *
   * @param from  the index at which to copy the first element from the specified collection.
   * @param other Collection to replace part of the receiver
   * @throws IndexOutOfBoundsException index is out of range (index &lt; 0 || index &gt;= size()).
   */
  @Override
  public void replaceFromWith(int from, Collection<Object[]> other) {
    checkRange(from, size);
    Iterator<Object[]> e = other.iterator();
    int index = from;
    int limit = Math.min(size - from, other.size());
    for (int i = 0; i < limit; i++) {
      elements[index++] = e.next();
    } //delta
  }

  /**
   * Retains (keeps) only the elements in the receiver that are contained in the specified other list. In other words,
   * removes from the receiver all of its elements that are not contained in the specified other list. Tests for
   * equality or identity as specified by <code>testForEquality</code>.
   *
   * @param other           the other list to test against.
   * @param testForEquality if <code>true</code> -> test for equality, otherwise for identity.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  public boolean retainAll(ObjectArrayList other, boolean testForEquality) {
    if (other.size == 0) {
      if (size == 0) {
        return false;
      }
      setSize(0);
      return true;
    }

    int limit = other.size - 1;
    int j = 0;
    Object[] theElements = elements;

    for (int i = 0; i < size; i++) {
      if (other.indexOfFromTo(theElements[i], 0, limit, testForEquality) >= 0) {
        theElements[j++] = theElements[i];
      }
    }

    boolean modified = (j != size);
    setSize(j);
    return modified;
  }

  /** Reverses the elements of the receiver. Last becomes first, second last becomes second first, and so on. */
  @Override
  public void reverse() {
    int limit = size / 2;
    int j = size - 1;

    Object[] theElements = elements;
    for (int i = 0; i < limit;) { //swap
      Object tmp = theElements[i];
      theElements[i++] = theElements[j];
      theElements[j--] = tmp;
    }
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   * @throws IndexOutOfBoundsException index is out of range (index &lt; 0 || index &gt;= size()).
   */
  public void set(int index, Object element) {
    if (index >= size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    elements[index] = element;
  }

  /**
   * Replaces the element at the specified position in the receiver with the specified element; <b>WARNING:</b> Does not
   * check preconditions. Provided with invalid parameters this method may access invalid indexes without throwing any
   * exception! <b>You should only use this method when you are absolutely sure that the index is within bounds.</b>
   * Precondition (unchecked): <tt>index &gt;= 0 && index &lt; size()</tt>.
   *
   * @param index   index of element to replace.
   * @param element element to be stored at the specified position.
   */
  public void setQuick(int index, Object element) {
    elements[index] = element;
  }

  /**
   * Randomly permutes the part of the receiver between <code>from</code> (inclusive) and <code>to</code> (inclusive).
   *
   * @param from the index of the first element (inclusive) to be permuted.
   * @param to   the index of the last element (inclusive) to be permuted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>size()&gt;0 && (from&lt;0 || from&gt;to ||
   *                                   to&gt;=size())</tt>).
   */
  @Override
  public void shuffleFromTo(int from, int to) {
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);

    org.apache.mahout.jet.random.Uniform gen =
        new org.apache.mahout.jet.random.Uniform(new org.apache.mahout.jet.random.engine.DRand(new java.util.Date()));
    Object[] theElements = elements;
    for (int i = from; i < to; i++) {
      int random = gen.nextIntFromTo(i, to);

      //swap(i, random)
      Object tmpElement = theElements[random];
      theElements[random] = theElements[i];
      theElements[i] = tmpElement;
    }
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
  public ObjectArrayList times(int times) {
    ObjectArrayList newList = new ObjectArrayList(times * size);
    for (int i = times; --i >= 0;) {
      newList.addAllOfFromTo(this, 0, size() - 1);
    }
    return newList;
  }

  /**
   * Returns an array containing all of the elements in the receiver in the correct order.  The runtime type of the
   * returned array is that of the specified array.  If the receiver fits in the specified array, it is returned
   * therein. Otherwise, a new array is allocated with the runtime type of the specified array and the size of the
   * receiver. <p> If the receiver fits in the specified array with room to spare (i.e., the array has more elements
   * than the receiver), the element in the array immediately following the end of the receiver is set to null.  This is
   * useful in determining the length of the receiver <em>only</em> if the caller knows that the receiver does not
   * contain any null elements.
   *
   * @param array the array into which the elements of the receiver are to be stored, if it is big enough; otherwise, a
   *              new array of the same runtime type is allocated for this purpose.
   * @return an array containing the elements of the receiver.
   * @throws ArrayStoreException the runtime type of <tt>array</tt> is not a supertype of the runtime type of every
   *                             element in the receiver.
   */
  public Object[] toArray(Object[] array) {
    if (array.length < size) {
      array = (Object[]) java.lang.reflect.Array.newInstance(array.getClass().getComponentType(), size);
    }

    Object[] theElements = elements;
    for (int i = size; --i >= 0;) {
      array[i] = theElements[i];
    }

    if (array.length > size) {
      array[size] = null;
    }

    return array;
  }

  /** Returns a <code>ArrayList</code> containing all the elements in the receiver. */
  @Override
  public List<Object[]> toList() {
    int mySize = size();
    Object[] theElements = elements;
    List<Object[]> list = new ArrayList<Object[]>(mySize);
    list.addAll(Arrays.<Object[]>asList(theElements).subList(0, mySize));
    return list;
  }

  /** Returns a string representation of the receiver, containing the String representation of each element. */
  public String toString() {
    return org.apache.mahout.matrix.Arrays.toString(partFromTo(0, size() - 1).elements());
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. Releases any superfluos internal memory. An
   * application can use this operation to minimize the storage of the receiver.
   */
  @Override
  public void trimToSize() {
    elements = org.apache.mahout.matrix.Arrays.trimToCapacity(elements, size());
  }
}
