/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix.list;

import org.apache.mahout.jet.math.Arithmetic;
import org.apache.mahout.matrix.bitvector.BitVector;
import org.apache.mahout.matrix.bitvector.QuickBitVector;
/**
 * Resizable compressed list holding numbers; based on the fact that a value in a given interval need not take more than <tt>log(max-min+1)</tt> bits; implemented with a <tt>org.apache.mahout.matrix.bitvector.BitVector</tt>.
 * First see the <a href="package-summary.html">package summary</a> and javadoc <a href="package-tree.html">tree view</a> to get the broad picture.
 * <p>
 * Numbers can be compressed when minimum and maximum of all values ever to be stored in the list are known.
 * For example, if min=16, max=27, only 4 bits are needed to store a value.
 * No compression is achieved for <tt>float</tt> and <tt>double</tt> values.
 * <p>
 * You can add, get and set elements quite similar to <tt>ArrayList</tt>.
 * <p>
 * <b>Applicability:</b> Applicable if the data is non floating point, highly skewed without "outliers" and minimum and maximum known in advance.
 * <p>
 * <b>Performance:</b> Basic operations like <tt>add()</tt>, <tt>get()</tt>, <tt>set()</tt>, <tt>size()</tt> and <tt>clear()</tt> are <tt>O(1)</tt>, i.e. run in constant time.
 * <dt>200Mhz Pentium Pro, JDK 1.2, NT:
 * <dt><tt>10^6</tt> calls to <tt>getQuick()</tt> --> <tt>0.5</tt> seconds.
 * (50 times slower than reading from a primitive array of the appropriate type.)
 * <dt><tt>10^6</tt> calls to <tt>setQuick()</tt> --> <tt>0.8</tt> seconds.
 * (15 times slower than writing to a primitive array of the appropriate type.)
 * <p>
 * This class can, for example, be useful when making large lists of numbers persistent.
 * Also useful when very large lists would otherwise consume too much main memory.
 * <p>
 * Upon instantiation a contract is signed that defines the interval values may fall into.
 * It is not legal to store values not contained in that interval.
 * WARNING: The contract is not checked. Be sure you do not store illegal values.
 * If you need to store <tt>float</tt> or <tt>double</tt> values, you must set the minimum and maximum to <tt>[Integer.MIN_VALUE,Integer.MAX_VALUE]</tt> or <tt>[Long.MIN_VALUE,Long.MAX_VALUE]</tt>, respectively.
 * <p>
 * Although access methods are only defined on <tt>long</tt> values you can also store
 * all other primitive data types: <tt>boolean</tt>, <tt>byte</tt>, <tt>short</tt>, <tt>int</tt>, <tt>long</tt>, <tt>float</tt>, <tt>double</tt> and <tt>char</tt>.
 * You can do this by explicitly representing them as <tt>long</tt> values.
 * Use casts for discrete data types.
 * Use the methods of <tt>java.lang.Float</tt> and <tt>java.lang.Double</tt> for floating point data types:
 * Recall that with those methods you can convert any floating point value to a <tt>long</tt> value and back <b>without losing any precision</b>:
 * <p>
 * <b>Example usage:</b><pre>
 * MinMaxNumberList list = ... instantiation goes here
 * double d1 = 1.234;
 * list.add(Double.doubleToLongBits(d1));
 * double d2 = Double.longBitsToDouble(list.get(0));
 * if (d1!=d2) log.info("This is impossible!");
 *
 * MinMaxNumberList list2 = ... instantiation goes here
 * float f1 = 1.234f;
 * list2.add((long) Float.floatToIntBits(f1));
 * float f2 = Float.intBitsToFloat((int)list2.get(0));
 * if (f1!=f2) log.info("This is impossible!");
 * </pre>
 *
 * @see LongArrayList
 * @see DistinctNumberList
 * @see java.lang.Float
 * @see java.lang.Double
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class MinMaxNumberList extends org.apache.mahout.matrix.list.AbstractLongList {

  protected long minValue;
  protected int bitsPerElement;
  protected long[] bits;
  protected int capacity;

  /**
   * Constructs an empty list with the specified initial capacity and the specified range of values allowed to be hold
   * in this list. Legal values are in the range [minimum,maximum], all inclusive.
   *
   * @param minimum         the minimum of values allowed to be hold in this list.
   * @param maximum         the maximum of values allowed to be hold in this list.
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  public MinMaxNumberList(long minimum, long maximum, int initialCapacity) {
    this.setUp(minimum, maximum, initialCapacity);
  }

  /**
   * Appends the specified element to the end of this list.
   *
   * @param element element to be appended to this list.
   */
  @Override
  public void add(long element) {
    // overridden for performance only.
    if (size == capacity) {
      ensureCapacity(size + 1);
    }
    int i = size * this.bitsPerElement;
    QuickBitVector.putLongFromTo(this.bits, element - this.minValue, i, i + this.bitsPerElement - 1);
    size++;
  }

  /**
   * Appends the elements <tt>elements[from]</tt> (inclusive), ..., <tt>elements[to]</tt> (inclusive) to the receiver.
   *
   * @param elements the elements to be appended to the receiver.
   * @param from     the index of the first element to be appended (inclusive)
   * @param to       the index of the last element to be appended (inclusive)
   */
  public void addAllOfFromTo(long[] elements, int from, int to) {
    // cache some vars for speed.
    int bitsPerElem = this.bitsPerElement;
    int bitsPerElemMinusOne = bitsPerElem - 1;
    long min = this.minValue;
    long[] theBits = this.bits;

    // now let's go.
    ensureCapacity(this.size + to - from + 1);
    int firstBit = this.size * bitsPerElem;
    int i = from;
    for (int times = to - from + 1; --times >= 0;) {
      QuickBitVector.putLongFromTo(theBits, elements[i++] - min, firstBit, firstBit + bitsPerElemMinusOne);
      firstBit += bitsPerElem;
    }
    this.size += (to - from + 1); //*bitsPerElem;
  }

  /** Returns the number of bits necessary to store a single element. */
  public int bitsPerElement() {
    return this.bitsPerElement;
  }

  /** Returns the number of bits necessary to store values in the range <tt>[minimum,maximum]</tt>. */
  public static int bitsPerElement(long minimum, long maximum) {
    int bits;
    if (1 + maximum - minimum > 0) {
      bits = (int) Math.round(Math.ceil(Arithmetic.log(2, 1 + maximum - minimum)));
    } else {
      // overflow or underflow in calculating "1+maximum-minimum"
      // happens if signed long representation is too short for doing unsigned calculations
      // e.g. if minimum==LONG.MIN_VALUE, maximum==LONG.MAX_VALUE
      // --> in such cases store all bits of values without any compression.
      bits = 64;
    }
    return bits;
  }

  /**
   * Ensures that the receiver can hold at least the specified number of elements without needing to allocate new
   * internal memory. If necessary, allocates new internal memory and increases the capacity of the receiver.
   *
   * @param minCapacity the desired minimum capacity.
   */
  @Override
  public void ensureCapacity(int minCapacity) {
    int oldCapacity = capacity;
    if (minCapacity > oldCapacity) {
      int newCapacity = (oldCapacity * 3) / 2 + 1;
      if (newCapacity < minCapacity) {
        newCapacity = minCapacity;
      }
      BitVector vector = toBitVector();
      vector.setSize(newCapacity * bitsPerElement);
      this.bits = vector.elements();
      this.capacity = newCapacity;
    }
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
    int i = index * this.bitsPerElement;
    return this.minValue + QuickBitVector.getLongFromTo(this.bits, i, i + this.bitsPerElement - 1);
  }

  /**
   * Copies all elements between index <tt>from</tt> (inclusive) and <tt>to</tt> (inclusive) into <tt>part</tt>,
   * starting at index <tt>partFrom</tt> within <tt>part</tt>. Elements are only copied if a corresponding flag within
   * <tt>qualificants</tt> is set.
   * More precisely:<pre>
   * for (; from<=to; from++, partFrom++, qualificantsFrom++) {
   *    if (qualificants==null || qualificants.get(qualificantsFrom)) {
   *       part[partFrom] = this.get(from);
   *    }
   * }
   * </pre>
   */
  public void partFromTo(int from, int to, BitVector qualificants, int qualificantsFrom, long[] part, int partFrom) {
    int width = to - from + 1;
    if (from < 0 || from > to || to >= size || qualificantsFrom < 0 ||
        (qualificants != null && qualificantsFrom + width > qualificants.size())) {
      throw new IndexOutOfBoundsException();
    }
    if (partFrom < 0 || partFrom + width > part.length) {
      throw new IndexOutOfBoundsException();
    }

    long minVal = this.minValue;
    int bitsPerElem = this.bitsPerElement;
    long[] theBits = this.bits;

    int q = qualificantsFrom;
    int p = partFrom;
    int j = from * bitsPerElem;

    //BitVector tmpBitVector = new BitVector(this.bits, this.size*bitsPerElem);
    for (int i = from; i <= to; i++, q++, p++, j += bitsPerElem) {
      if (qualificants == null || qualificants.get(q)) {
        //part[p] = minVal + tmpBitVector.getLongFromTo(j, j+bitsPerElem-1);
        part[p] = minVal + QuickBitVector.getLongFromTo(theBits, j, j + bitsPerElem - 1);
      }
    }
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
    int i = index * this.bitsPerElement;
    QuickBitVector.putLongFromTo(this.bits, element - this.minValue, i, i + this.bitsPerElement - 1);
  }

  /**
   * Sets the size of the receiver without modifying it otherwise. This method should not release or allocate new memory
   * but simply set some instance variable like <tt>size</tt>.
   */
  @Override
  protected void setSizeRaw(int newSize) {
    super.setSizeRaw(newSize);
  }

  /**
   * Sets the receiver to an empty list with the specified initial capacity and the specified range of values allowed to
   * be hold in this list. Legal values are in the range [minimum,maximum], all inclusive.
   *
   * @param minimum         the minimum of values allowed to be hold in this list.
   * @param maximum         the maximum of values allowed to be hold in this list.
   * @param initialCapacity the number of elements the receiver can hold without auto-expanding itself by allocating new
   *                        internal memory.
   */
  protected void setUp(long minimum, long maximum, int initialCapacity) {
    setUpBitsPerEntry(minimum, maximum);

    //this.capacity=initialCapacity;
    this.bits = QuickBitVector.makeBitVector(initialCapacity, this.bitsPerElement);
    this.capacity = initialCapacity;
    this.size = 0;
  }

  /** This method was created in VisualAge. */
  protected void setUpBitsPerEntry(long minimum, long maximum) {
    this.bitsPerElement = bitsPerElement(minimum, maximum);
    if (this.bitsPerElement != 64) {
      this.minValue = minimum;
      // overflow or underflow in calculating "1+maxValue-minValue"
      // happens if signed long representation is too short for doing unsigned calculations
      // e.g. if minValue==LONG.MIN_VALUE, maxValue=LONG.MAX_VALUE
      // --> in such cases store all bits of values without any en/decoding
    } else {
      this.minValue = 0;
    }
  }

  /**
   * Returns the receiver seen as bitvector. WARNING: The bitvector and the receiver share the backing bits. Modifying
   * one of them will affect the other.
   */
  public BitVector toBitVector() {
    return new BitVector(this.bits, this.capacity * bitsPerElement);
  }

  /**
   * Trims the capacity of the receiver to be the receiver's current size. An application can use this operation to
   * minimize the storage of the receiver.
   */
  @Override
  public void trimToSize() {
    int oldCapacity = capacity;
    if (size < oldCapacity) {
      BitVector vector = toBitVector();
      vector.setSize(size);
      this.bits = vector.elements();
      this.capacity = size;
    }
  }

  /**
   * deprecated Returns the minimum element legal to the stored in the receiver. Remark: This does not mean that such a
   * minimum element is currently contained in the receiver.
   *
   * @deprecated
   */
  @Deprecated
  public long xminimum() {
    return this.minValue;
  }
}
