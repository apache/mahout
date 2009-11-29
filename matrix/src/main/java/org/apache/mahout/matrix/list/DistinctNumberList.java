/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.list;

/**
 * Resizable compressed list holding numbers; based on the fact that a number from a large list with few distinct values need not take more than <tt>log(distinctValues)</tt> bits; implemented with a <tt>MinMaxNumberList</tt>.
 * First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * This class can, for example, be useful when making large lists of numbers persistent.
 * Also useful when very large lists would otherwise consume too much main memory.
 * <p>
 * You can add, get and set elements quite similar to <tt>ArrayList</tt>.
 * <p>
 * <b>Applicability:</b> Applicable if data is highly skewed and legal values are known in advance. Robust in the presence of "outliers".
 * <p>
 * <b>Performance:</b> Operations <tt>get()</tt>, <tt>size()</tt> and <tt>clear()</tt> are <tt>O(1)</tt>, i.e. run in constant time.
 * Operations like <tt>add()</tt> and <tt>set()</tt> are <tt>O(log(distinctValues.length))</tt>.
 * <p>
 * Upon instantiation a contract is signed that defines the distinct values allowed to be hold in this list.
 * It is not legal to store elements other than specified by the contract.
 * Any attempt to violate the contract will throw an <tt>IllegalArgumentException</tt>.
 * <p>
 * Although access methods are only defined on <tt>long</tt> values you can also store
 * all other primitive data types: <tt>boolean</tt>, <tt>byte</tt>, <tt>short</tt>, <tt>int</tt>, <tt>long</tt>, <tt>float</tt>, <tt>double</tt> and <tt>char</tt>.
 * You can do this by explicitly representing them as <tt>long</tt> values.
 * Use casts for discrete data types.
 * Use the methods of <tt>java.lang.Float</tt> and <tt>java.lang.Double</tt> for floating point data types:
 * Recall that with those methods you can convert any floating point value to a <tt>long</tt> value and back <b>without losing any precision</b>:
 * <p>
 * <b>Example usage:</b><pre>
 * DistinctNumberList list = ... instantiation goes here
 * double d1 = 1.234;
 * list.add(Double.doubleToLongBits(d1));
 * double d2 = Double.longBitsToDouble(list.get(0));
 * if (d1!=d2) log.info("This is impossible!");
 *
 * DistinctNumberList list2 = ... instantiation goes here
 * float f1 = 1.234f;
 * list2.add((long) Float.floatToIntBits(f1));
 * float f2 = Float.intBitsToFloat((int)list2.get(0));
 * if (f1!=f2) log.info("This is impossible!");
 * </pre>
 *
 * @see LongArrayList
 * @see MinMaxNumberList
 * @see java.lang.Float
 * @see java.lang.Double
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class DistinctNumberList extends AbstractLongList {

  private long[] distinctValues;
  private MinMaxNumberList elements;

  /**
   * Constructs an empty list with the specified initial capacity and the specified distinct values allowed to be hold
   * in this list.
   *
   * @param distinctValues  an array sorted ascending containing the distinct values allowed to be hold in this list.
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  public DistinctNumberList(long[] distinctValues, int initialCapacity) {
    setUp(distinctValues, initialCapacity);
  }

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  @Override
  public void add(long element) {
    //overridden for performance only.
    elements.add(codeOf(element));
    size++;
  }

  /** Returns the code that shall be stored for the given element. */
  protected int codeOf(long element) {
    int index = java.util.Arrays.binarySearch(distinctValues, element);
    if (index < 0) {
      throw new IllegalArgumentException("Element=" + element + " not contained in distinct elements.");
    }
    return index;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    elements.ensureCapacity(minCapacity);
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
  public long getQuick(int index) {
    return distinctValues[(int) (elements.getQuick(index))];
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
    elements.removeFromTo(from, to);
    size -= to - from + 1;
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
  public void setQuick(int index, long element) {
    elements.setQuick(index, codeOf(element));
  }

  /**
   * Sets the size of the receiver without modifying it otherwise. This method should not release or allocate new memory
   * but simply set some instance variable like <tt>size</tt>.
   */
  @Override
  protected void setSizeRaw(int newSize) {
    super.setSizeRaw(newSize);
    elements.setSizeRaw(newSize);
  }

  /**
   * Sets the receiver to an empty list with the specified initial capacity and the specified distinct values allowed to
   * be hold in this list.
   *
   * @param distinctValues  an array sorted ascending containing the distinct values allowed to be hold in this list.
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  protected void setUp(long[] distinctValues, int initialCapacity) {
    this.distinctValues = distinctValues;
    //java.util.Arrays.sort(this.distinctElements);
    this.elements = new MinMaxNumberList(0, distinctValues.length - 1, initialCapacity);
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. An application can use this operation to
   * minimize the storage of the receiver.
   */
  @Override
  public void trimToSize() {
    elements.trimToSize();
  }
}
