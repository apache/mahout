/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.list;

import org.apache.mahout.jet.random.Uniform;
import org.apache.mahout.jet.random.engine.DRand;
import org.apache.mahout.matrix.function.DoubleProcedure;

import java.util.Date;
/**
 Resizable list holding <code>double</code> elements; implemented with arrays.
 First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DoubleArrayList extends AbstractDoubleList {

  /**
   * The array buffer into which the elements of the list are stored. The capacity of the list is the length of this
   * array buffer.
   */
  private double[] elements;

  /** Constructs an empty list. */
  public DoubleArrayList() {
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
  public DoubleArrayList(double[] elements) {
    elements(elements);
  }

  /**
   * Constructs an empty list with the specified initial capacity.
   *
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  public DoubleArrayList(int initialCapacity) {
    this(new double[initialCapacity]);
    setSizeRaw(0);
  }

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  @Override
  public void add(double element) {
    // overridden for performance only.
    if (size == elements.length) {
      ensureCapacity(size + 1);
    }
    elements[size++] = element;
  }

  /**
   * Inserts the specified element before the specified position into the receiver. Shifts the element currently at that
   * position (if any) and any subsequent elements to the right.
   *
   * @param index   index before which the specified element is to be inserted (must be in [0,size]).
   * @param element element to be inserted.
   * @throws IndexOutOfBoundsException index is out of range (<tt>index &lt; 0 || index &gt; size()</tt>).
   */
  @Override
  public void beforeInsert(int index, double element) {
    // overridden for performance only.
    if (size == index) {
      add(element);
      return;
    }
    if (index > size || index < 0) {
      throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
    }
    ensureCapacity(size + 1);
    System.arraycopy(elements, index, elements, index + 1, size - index);
    elements[index] = element;
    size++;
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
   * @see org.apache.mahout.matrix.Sorting
   * @see java.util.Arrays
   */
  @Override
  public int binarySearchFromTo(double key, int from, int to) {
    return org.apache.mahout.matrix.Sorting.binarySearchFromTo(this.elements, key, from, to);
  }

  /**
   * Returns a deep copy of the receiver.
   *
   * @return a deep copy of the receiver.
   */
  @Override
  public Object clone() {
    // overridden for performance only.
    DoubleArrayList clone = new DoubleArrayList(elements.clone());
    clone.setSizeRaw(size);
    return clone;
  }

  /**
   * Returns a deep copy of the receiver; uses <code>clone()</code> and casts the result.
   *
   * @return a deep copy of the receiver.
   */
  public DoubleArrayList copy() {
    return (DoubleArrayList) clone();
  }

  /**
   * Returns the elements currently stored, including invalid elements between size and capacity, if any.
   *
   * <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the array is not copied</b>. So if
   * subsequently you modify the returned array directly via the [] operator, be sure you know what you're doing.
   *
   * @return the elements currently stored.
   */
  @Override
  public double[] elements() {
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
  @Override
  public AbstractDoubleList elements(double[] elements) {
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
  @Override
  public void ensureCapacity(int minCapacity) {
    elements = org.apache.mahout.matrix.Arrays.ensureCapacity(elements, minCapacity);
  }

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
    // overridden for performance only.
    if (!(otherObj instanceof DoubleArrayList)) {
      return super.equals(otherObj);
    }
    if (this == otherObj) {
      return true;
    }
    if (otherObj == null) {
      return false;
    }
    DoubleArrayList other = (DoubleArrayList) otherObj;
    if (size() != other.size()) {
      return false;
    }

    double[] theElements = elements();
    double[] otherElements = other.elements();
    for (int i = size(); --i >= 0;) {
      if (theElements[i] != otherElements[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Applies a procedure to each element of the receiver, if any. Starts at index 0, moving rightwards.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   */
  @Override
  public boolean forEach(DoubleProcedure procedure) {
    // overridden for performance only.
    double[] theElements = elements;
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
  @Override
  public double get(int index) {
    // overridden for performance only.
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
  @Override
  public double getQuick(int index) {
    return elements[index];
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
  @Override
  public int indexOfFromTo(double element, int from, int to) {
    // overridden for performance only.
    if (size == 0) {
      return -1;
    }
    checkRangeFromTo(from, to, size);

    double[] theElements = elements;
    for (int i = from; i <= to; i++) {
      if (element == theElements[i]) {
        return i;
      } //found
    }
    return -1; //not found
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
  @Override
  public int lastIndexOfFromTo(double element, int from, int to) {
    // overridden for performance only.
    if (size == 0) {
      return -1;
    }
    checkRangeFromTo(from, to, size);

    double[] theElements = elements;
    for (int i = to; i >= from; i--) {
      if (element == theElements[i]) {
        return i;
      } //found
    }
    return -1; //not found
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
  @Override
  public AbstractDoubleList partFromTo(int from, int to) {
    if (size == 0) {
      return new DoubleArrayList(0);
    }

    checkRangeFromTo(from, to, size);

    double[] part = new double[to - from + 1];
    System.arraycopy(elements, from, part, 0, to - from + 1);
    return new DoubleArrayList(part);
  }

  /**
   * Removes from the receiver all elements that are contained in the specified list. Tests for identity.
   *
   * @param other the other list.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  @Override
  public boolean removeAll(AbstractDoubleList other) {
    // overridden for performance only.
    if (!(other instanceof DoubleArrayList)) {
      return super.removeAll(other);
    }

    /* There are two possibilities to do the thing
       a) use other.indexOf(...)
       b) sort other, then use other.binarySearch(...)

       Let's try to figure out which one is faster. Let M=size, N=other.size, then
       a) takes O(M*N) steps
       b) takes O(N*logN + M*logN) steps (sorting is O(N*logN) and binarySearch is O(logN))

       Hence, if N*logN + M*logN < M*N, we use b) otherwise we use a).
    */
    if (other.isEmpty()) {
      return false;
    } //nothing to do
    int limit = other.size() - 1;
    int j = 0;
    double[] theElements = elements;
    int mySize = size();

    double N = (double) other.size();
    double M = (double) mySize;
    if ((N + M) * org.apache.mahout.jet.math.Arithmetic.log2(N) < M * N) {
      // it is faster to sort other before searching in it
      DoubleArrayList sortedList = (DoubleArrayList) other.clone();
      sortedList.quickSort();

      for (int i = 0; i < mySize; i++) {
        if (sortedList.binarySearchFromTo(theElements[i], 0, limit) < 0) {
          theElements[j++] = theElements[i];
        }
      }
    } else {
      // it is faster to search in other without sorting
      for (int i = 0; i < mySize; i++) {
        if (other.indexOfFromTo(theElements[i], 0, limit) < 0) {
          theElements[j++] = theElements[i];
        }
      }
    }

    boolean modified = (j != mySize);
    setSize(j);
    return modified;
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
  @Override
  public void replaceFromToWithFrom(int from, int to, AbstractDoubleList other, int otherFrom) {
    // overridden for performance only.
    if (!(other instanceof DoubleArrayList)) {
      // slower
      super.replaceFromToWithFrom(from, to, other, otherFrom);
      return;
    }
    int length = to - from + 1;
    if (length > 0) {
      checkRangeFromTo(from, to, size());
      checkRangeFromTo(otherFrom, otherFrom + length - 1, other.size());
      System.arraycopy(((DoubleArrayList) other).elements, otherFrom, elements, from, length);
    }
  }

  /**
   * Retains (keeps) only the elements in the receiver that are contained in the specified other list. In other words,
   * removes from the receiver all of its elements that are not contained in the specified other list.
   *
   * @param other the other list to test against.
   * @return <code>true</code> if the receiver changed as a result of the call.
   */
  @Override
  public boolean retainAll(AbstractDoubleList other) {
    // overridden for performance only.
    if (!(other instanceof DoubleArrayList)) {
      return super.retainAll(other);
    }

    /* There are two possibilities to do the thing
       a) use other.indexOf(...)
       b) sort other, then use other.binarySearch(...)

       Let's try to figure out which one is faster. Let M=size, N=other.size, then
       a) takes O(M*N) steps
       b) takes O(N*logN + M*logN) steps (sorting is O(N*logN) and binarySearch is O(logN))

       Hence, if N*logN + M*logN < M*N, we use b) otherwise we use a).
    */
    int limit = other.size() - 1;
    int j = 0;
    double[] theElements = elements;
    int mySize = size();

    double N = (double) other.size();
    double M = (double) mySize;
    if ((N + M) * org.apache.mahout.jet.math.Arithmetic.log2(N) < M * N) {
      // it is faster to sort other before searching in it
      DoubleArrayList sortedList = (DoubleArrayList) other.clone();
      sortedList.quickSort();

      for (int i = 0; i < mySize; i++) {
        if (sortedList.binarySearchFromTo(theElements[i], 0, limit) >= 0) {
          theElements[j++] = theElements[i];
        }
      }
    } else {
      // it is faster to search in other without sorting
      for (int i = 0; i < mySize; i++) {
        if (other.indexOfFromTo(theElements[i], 0, limit) >= 0) {
          theElements[j++] = theElements[i];
        }
      }
    }

    boolean modified = (j != mySize);
    setSize(j);
    return modified;
  }

  /** Reverses the elements of the receiver. Last becomes first, second last becomes second first, and so on. */
  @Override
  public void reverse() {
    // overridden for performance only.
    int limit = size / 2;
    int j = size - 1;

    double[] theElements = elements;
    for (int i = 0; i < limit;) { //swap
      double tmp = theElements[i];
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
  @Override
  public void set(int index, double element) {
    // overridden for performance only.
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
  @Override
  public void setQuick(int index, double element) {
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
    // overridden for performance only.
    if (size == 0) {
      return;
    }
    checkRangeFromTo(from, to, size);

    Uniform gen = new Uniform(new DRand(new Date()));
    double[] theElements = elements;
    for (int i = from; i < to; i++) {
      int random = gen.nextIntFromTo(i, to);

      //swap(i, random)
      double tmpElement = theElements[random];
      theElements[random] = theElements[i];
      theElements[i] = tmpElement;
    }
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
