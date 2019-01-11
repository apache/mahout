/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.list;

/**
 Resizable list holding <code>long</code> elements; implemented with arrays; not efficient; just to
 demonstrate which methods you must override to implement a fully functional list.
 */
public class SimpleLongArrayList extends AbstractLongList {

  /**
   * The array buffer into which the elements of the list are stored. The capacity of the list is the length of this
   * array buffer.
   */
  private long[] elements;

  /** Constructs an empty list. */
  public SimpleLongArrayList() {
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
  public SimpleLongArrayList(long[] elements) {
    elements(elements);
  }

  /**
   * Constructs an empty list with the specified initial capacity.
   *
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  private SimpleLongArrayList(int initialCapacity) {
    if (initialCapacity < 0) {
      throw new IllegalArgumentException("Illegal Capacity: " + initialCapacity);
    }

    this.elements(new long[initialCapacity]);
    size = 0;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    elements = org.apache.mahout.math.Arrays.ensureCapacity(elements, minCapacity);
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
  protected long getQuick(int index) {
    return elements[index];
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
  protected void setQuick(int index, long element) {
    elements[index] = element;
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. An application can use this operation to
   * minimize the storage of the receiver.
   */
  @Override
  public void trimToSize() {
    elements = org.apache.mahout.math.Arrays.trimToCapacity(elements, size());
  }
}
