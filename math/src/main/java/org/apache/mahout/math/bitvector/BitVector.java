/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.bitvector;

import org.apache.mahout.math.PersistentObject;
import org.apache.mahout.math.function.IntProcedure;

/**
 * Fixed sized (non resizable) bitvector.
 * Upon instance construction a bitvector is told to hold a fixed number of bits - it's size.
 * The size can be any number (need not be a power of 2 or so).
 * The bits of a <tt>BitVector</tt> are indexed by nonnegative integers. 
 * Any attempt to access a bit at an <tt>index&lt;0 || index&gt;=size()</tt> will throw an <tt>IndexOutOfBoundsException</tt>.
 * <p>
 * Individual indexed bits can be examined, set, or cleared.
 * Subranges can quickly be extracted, copied and replaced.
 * Quick iteration over subranges is provided by optimized internal iterators (<tt>forEach()</tt> methods).
 * One <code>BitVector</code> may be used to modify the contents of another 
 * <code>BitVector</code> through logical AND, OR, XOR and other similar operations.
 * <p>
 * All operations consider the bits <tt>0..size()-1</tt> and nothing else.
 * Operations involving two bitvectors (like AND, OR, XOR, etc.) will throw an <tt>IllegalArgumentException</tt> if the secondary bit vector has a size smaller than the receiver.
 * <p>
 * A <tt>BitVector</tt> is never automatically resized,
 * but it can manually be grown or shrinked via <tt>setSize(...)</tt>.
 * <p>
 * For use cases that need to store several bits per information entity, quick accessors are provided that interpret subranges as 64 bit <tt>long</tt> integers.
 * <p>
 * Why this class? Fist, <tt>boolean[]</tt> take one byte per stored bit. This class takes one bit per stored bit.
 * Second, many applications find the semantics of <tt>java.util.BitSet</tt> not particularly helpful for their needs.
 * Third, operations working on all bits of a bitvector are extremely quick.
 * For example, on NT, Pentium Pro 200 Mhz, SunJDK1.2.2, java -classic, for two bitvectors A,B (both much larger than processor cache), the following results are obtained.
 * <ul>
 * <li><tt>A.and(B)</tt> i.e. A = A & B --> runs at about 35 MB/sec
 * <li><tt>A.cardinality()</tt>, i.e. determining the selectivity, the number of bits in state "true" --> runs at about 80 MB/sec
 * <li>Similar performance for <tt>or, xor, andNot, not, copy, replace, partFromTo, indexOf, clear</tt> etc.
 * </ul>
 * If you need extremely quick access to individual bits: Although getting and setting individual bits with methods <tt>get(...)</tt>, <tt>set(...)</tt> and <tt>put(...)</tt>is quick, it is even quicker (<b>but not safe</b>) to use <tt>getQuick(...)</tt> and <tt>putQuick(...)</tt> or even <tt>QuickBitVector</tt>.
 * <p>
 * <b>Note</b> that this implementation is not synchronized.
 *
 * @see     QuickBitVector
 * @see     BitMatrix
 * @see     java.util.BitSet
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class BitVector extends PersistentObject {
  /*
   * Bits are packed into arrays of "units."  Currently a unit is a long,
   * which consists of 64 bits, requiring 6 address bits.  The choice of unit
   * is determined purely by performance concerns.
   */

  /**
   * The bits of this object.  The ith bit is stored in bits[i/64] at bit position i % 64 (where bit position 0 refers
   * to the least significant bit and 63 refers to the most significant bit).
   */
  private long[] bits;

  private int nbits; //the size

  // IntProcedure for method indexOfFromTo(...)

  private static class IndexProcedure implements IntProcedure {

    private int foundPos = -1;

    @Override
    public boolean apply(int index) {
      foundPos = index;
      return false;
    }
  }

  /**
   * You normally need not use this method. Use this method only if performance is critical. Constructs a bit vector
   * with the given backing bits and size. <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the
   * array is not copied</b>. So if subsequently you modify the specified array directly via the [] operator, be sure
   * you know what you're doing.
   *
   * <p>A bitvector is modelled as a long array, i.e. <tt>long[] bits</tt> holds bits of a bitvector. Each long value
   * holds 64 bits. The i-th bit is stored in bits[i/64] at bit position i % 64 (where bit position 0 refers to the
   * least significant bit and 63 refers to the most significant bit).
   *
   * @throws IllegalArgumentException if <tt>size &lt; 0 || size &gt; bits.length*64</tt>.
   */
  public BitVector(long[] bits, int size) {
    elements(bits, size);
  }

  /**
   * Constructs a bit vector that holds <tt>size</tt> bits. All bits are initially <tt>false</tt>.
   *
   * @param size the number of bits the bit vector shall have.
   * @throws IllegalArgumentException if <tt>size &lt; 0</tt>.
   */
  public BitVector(int size) {
    this(QuickBitVector.makeBitVector(size, 1), size);
  }

  /**
   * Performs a logical <b>AND</b> of the receiver with another bit vector (A = A & B). The receiver is modified so that
   * a bit in it has the value <code>true</code> if and only if it already had the value <code>true</code> and the
   * corresponding bit in the other bit vector argument has the value <code>true</code>.
   *
   * @param other a bit vector.
   * @throws IllegalArgumentException if <tt>size() &gt; other.size()</tt>.
   */
  public void and(BitVector other) {
    if (this == other) {
      return;
    }
    checkSize(other);
    long[] theBits = this.bits; // cached for speed.
    long[] otherBits = other.bits; //cached for speed.
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] &= otherBits[i];
    }
  }

  /**
   * Clears all of the bits in receiver whose corresponding bit is set in the other bitvector (A = A \ B). In other
   * words, determines the difference (A=A\B) between two bitvectors.
   *
   * @param other a bitvector with which to mask the receiver.
   * @throws IllegalArgumentException if <tt>size() &gt; other.size()</tt>.
   */
  public void andNot(BitVector other) {
    checkSize(other);
    long[] theBits = this.bits; // cached for speed.
    long[] otherBits = other.bits; //cached for speed.
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] &= ~otherBits[i];
    }
  }

  /**
   * Returns the number of bits currently in the <tt>true</tt> state. Optimized for speed. Particularly quick if the
   * receiver is either sparse or dense.
   */
  public int cardinality() {
    int cardinality = 0;
    int fullUnits = numberOfFullUnits();
    int bitsPerUnit = QuickBitVector.BITS_PER_UNIT;

    // determine cardinality on full units
    long[] theBits = bits;
    for (int i = fullUnits; --i >= 0;) {
      long val = theBits[i];
      if (val == -1L) { // all bits set?
        cardinality += bitsPerUnit;
      } else if (val != 0L) { // more than one bit set?
        for (int j = bitsPerUnit; --j >= 0;) {
          if ((val & (1L << j)) != 0) {
            cardinality++;
          }
        }
      }
    }

    // determine cardinality on remaining partial unit, if any.
    for (int j = numberOfBitsInPartialUnit(); --j >= 0;) {
      if ((theBits[fullUnits] & (1L << j)) != 0) {
        cardinality++;
      }
    }

    return cardinality;
  }

  /** Checks if the given range is within the contained array's bounds. */
  private static void checkRangeFromTo(int from, int to, int theSize) {
    if (from < 0 || from > to || to >= theSize) {
      throw new IndexOutOfBoundsException("from: " + from + ", to: " + to + ", size=" + theSize);
    }
  }

  /** Sanity check for operations requiring another bitvector with at least the same size. */
  protected void checkSize(BitVector other) {
    if (nbits > other.size()) {
      throw new IllegalArgumentException("Incompatible sizes: size=" + nbits + ", other.size()=" + other.size());
    }
  }

  /** Clears all bits of the receiver. */
  public void clear() {
    long[] theBits = this.bits;
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] = 0L;
    }

    //new LongArrayList(bits).fillFromToWith(0,size()-1,0L);
  }

  /**
   * Changes the bit with index <tt>bitIndex</tt> to the "clear" (<tt>false</tt>) state.
   *
   * @param bitIndex the index of the bit to be cleared.
   * @throws IndexOutOfBoundsException if <tt>bitIndex&lt;0 || bitIndex&gt;=size()</tt>
   */
  public void clear(int bitIndex) {
    if (bitIndex < 0 || bitIndex >= nbits) {
      throw new IndexOutOfBoundsException(String.valueOf(bitIndex));
    }
    QuickBitVector.clear(bits, bitIndex);
  }

  /**
   * Cloning this <code>BitVector</code> produces a new <code>BitVector</code> that is equal to it. The clone of the bit
   * vector is another bit vector that has exactly the same bits set to <code>true</code> as this bit vector and the
   * same current size, but independent state.
   *
   * @return a deep copy of this bit vector.
   */
  @Override
  public Object clone() {
    BitVector clone = (BitVector) super.clone();
    if (this.bits != null) {
      clone.bits = this.bits.clone();
    }
    return clone;
  }

  /**
   * Returns a deep copy of the receiver; calls <code>clone()</code> and casts the result.
   *
   * @return a deep copy of the receiver.
   */
  public BitVector copy() {
    return (BitVector) clone();
  }

  /**
   * You normally need not use this method. Use this method only if performance is critical. Returns the bit vector's
   * backing bits. <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the array is not copied</b>.
   * So if subsequently you modify the returned array directly via the [] operator, be sure you know what you're doing.
   *
   * <p>A bitvector is modelled as a long array, i.e. <tt>long[] bits</tt> holds bits of a bitvector. Each long value
   * holds 64 bits. The i-th bit is stored in bits[i/64] at bit position i % 64 (where bit position 0 refers to the
   * least significant bit and 63 refers to the most significant bit).
   */
  public long[] elements() {
    return bits;
  }

  /**
   * You normally need not use this method. Use this method only if performance is critical. Sets the bit vector's
   * backing bits and size. <b>WARNING:</b> For efficiency reasons and to keep memory usage low, <b>the array is not
   * copied</b>. So if subsequently you modify the specified array directly via the [] operator, be sure you know what
   * you're doing.
   *
   * <p>A bitvector is modelled as a long array, i.e. <tt>long[] bits</tt> holds bits of a bitvector. Each long value
   * holds 64 bits. The i-th bit is stored in bits[i/64] at bit position i % 64 (where bit position 0 refers to the
   * least significant bit and 63 refers to the most significant bit).
   *
   * @param bits the backing bits of the bit vector.
   * @param size the number of bits the bit vector shall hold.
   * @throws IllegalArgumentException if <tt>size &lt; 0 || size &gt; bits.length*64</tt>.
   */
  public void elements(long[] bits, int size) {
    if (size < 0 || size > bits.length * QuickBitVector.BITS_PER_UNIT) {
      throw new IllegalArgumentException();
    }
    this.bits = bits;
    this.nbits = size;
  }

  /**
   * Compares this object against the specified object. The result is <code>true</code> if and only if the argument is
   * not <code>null</code> and is a <code>BitVector</code> object that has the same size as the receiver and the same
   * bits set to <code>true</code> as the receiver. That is, for every nonnegative <code>int</code> index
   * <code>k</code>,
   * <pre>((BitVector)obj).get(k) == this.get(k)</pre>
   * must be true.
   *
   * @param obj the object to compare with.
   * @return <code>true</code> if the objects are the same; <code>false</code> otherwise.
   */
  public boolean equals(Object obj) {
    if (obj == null || !(obj instanceof BitVector)) {
      return false;
    }
    if (this == obj) {
      return true;
    }

    BitVector other = (BitVector) obj;
    if (size() != other.size()) {
      return false;
    }

    int fullUnits = numberOfFullUnits();
    // perform logical comparison on full units
    for (int i = fullUnits; --i >= 0;) {
      if (bits[i] != other.bits[i]) {
        return false;
      }
    }

    // perform logical comparison on remaining bits
    int i = fullUnits * QuickBitVector.BITS_PER_UNIT;
    for (int times = numberOfBitsInPartialUnit(); --times >= 0;) {
      if (get(i) != other.get(i)) {
        return false;
      }
      i++;
    }

    return true;
  }

  /**
   * Applies a procedure to each bit index within the specified range that holds a bit in the given state. Starts at
   * index <tt>from</tt>, moves rightwards to <tt>to</tt>. Useful, for example, if you want to copy bits into an image
   * or somewhere else. <p> Optimized for speed. Particularly quick if one of the following conditions holds <ul>
   * <li><tt>state==true</tt> and the receiver is sparse (<tt>cardinality()</tt> is small compared to <tt>size()</tt>).
   * <li><tt>state==false</tt> and the receiver is dense (<tt>cardinality()</tt> is large compared to <tt>size()</tt>).
   * </ul>
   *
   * @param from      the leftmost search position, inclusive.
   * @param to        the rightmost search position, inclusive.
   * @param state     element to search for.
   * @param procedure a procedure object taking as argument the current bit index. Stops iteration if the procedure
   *                  returns <tt>false</tt>, otherwise continues.
   * @return <tt>false</tt> if the procedure stopped before all elements where iterated over, <tt>true</tt> otherwise.
   * @throws IndexOutOfBoundsException if (<tt>size()&gt;0 && (from&lt;0 || from&gt;to || to&gt;=size())</tt>).
   */
  public boolean forEachIndexFromToInState(int from, int to, boolean state,
                                           IntProcedure procedure) {
    /*
    // this version is equivalent to the low level version below, but about 100 times slower for large ranges.
    if (nbits==0) return true;
    checkRangeFromTo(from, to, nbits);
    final long[] theBits = this.bits; // cached for speed
    int length=to-from+1;
    for (int i=from; --length >= 0; i++) {
      if (QuickBitVector.get(theBits,i)==state) {
        if (!function.apply(i)) return false;
      }
    }
    return true;
    */


    /*
    * This low level implementation exploits the fact that for any full unit one can determine in O(1)
    * whether it contains at least one true bit,
    * and whether it contains at least one false bit.
    * Thus, 64 bits can often be skipped with one simple comparison if the vector is either sparse or dense.
    *
    * However, careful coding must be done for leading and/or trailing units which are not entirely contained in the query range.
    */

    if (nbits == 0) {
      return true;
    }
    checkRangeFromTo(from, to, nbits);

    // Cache some vars for speed.
    long[] theBits = this.bits;
    int bitsPerUnit = QuickBitVector.BITS_PER_UNIT;

    // Prepare
    int fromUnit = QuickBitVector.unit(from);
    int toUnit = QuickBitVector.unit(to);
    int i = from; // current bitvector index

    // Iterate over the leading partial unit, if any.
    int bitIndex = QuickBitVector.offset(from);
    int partialWidth;
    if (bitIndex > 0) { // There exists a leading partial unit.
      partialWidth = Math.min(to - from + 1, bitsPerUnit - bitIndex);
      for (; --partialWidth >= 0; i++) {
        if (QuickBitVector.get(theBits, i) == state) {
          if (!procedure.apply(i)) {
            return false;
          }
        }
      }
      fromUnit++; // leading partial unit is done.
    }

    if (i > to) {
      return true;
    } // done

    // If there is a trailing partial unit, then there is one full unit less to test.
    bitIndex = QuickBitVector.offset(to);
    if (bitIndex < bitsPerUnit - 1) {
      toUnit--; // trailing partial unit needs to be tested extra.
      partialWidth = bitIndex + 1;
    } else {
      partialWidth = 0;
    }

    // Iterate over all full units, if any.
    // (It does not matter that iterating over partial units is a little bit slow,
    // the only thing that matters is that iterating over full units is quick.)
    long comparator;
    if (state) {
      comparator = 0L;
    } else {
      comparator = ~0L;
    } // all 64 bits set

    for (int unit = fromUnit; unit <= toUnit; unit++) {
      long val = theBits[unit];
      if (val != comparator) {
        // at least one element within current unit matches.
        // iterate over all bits within current unit.
        if (state) {
          for (int j = 0, k = bitsPerUnit; --k >= 0; i++) {
            if ((val & (1L << j++)) != 0L) { // is bit set?
              if (!procedure.apply(i)) {
                return false;
              }
            }
          }
        } else {
          for (int j = 0, k = bitsPerUnit; --k >= 0; i++) {
            if ((val & (1L << j++)) == 0L) { // is bit cleared?
              if (!procedure.apply(i)) {
                return false;
              }
            }
          }
        }
      } else {
        i += bitsPerUnit;
      }
    }

    // Iterate over trailing partial unit, if any.
    for (; --partialWidth >= 0; i++) {
      if (QuickBitVector.get(theBits, i) == state) {
        if (!procedure.apply(i)) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Returns from the bitvector the value of the bit with the specified index. The value is <tt>true</tt> if the bit
   * with the index <tt>bitIndex</tt> is currently set; otherwise, returns <tt>false</tt>.
   *
   * @param bitIndex the bit index.
   * @return the value of the bit with the specified index.
   * @throws IndexOutOfBoundsException if <tt>bitIndex&lt;0 || bitIndex&gt;=size()</tt>
   */
  public boolean get(int bitIndex) {
    if (bitIndex < 0 || bitIndex >= nbits) {
      throw new IndexOutOfBoundsException(String.valueOf(bitIndex));
    }
    return QuickBitVector.get(bits, bitIndex);
  }

  /**
   * Returns a long value representing bits of the receiver from index <tt>from</tt> to index <tt>to</tt>. Bits are
   * returned as a long value with the return value having bit 0 set to bit <code>from</code>, ..., bit
   * <code>to-from</code> set to bit <code>to</code>. All other bits of the return value are set to 0. If
   * <tt>to-from+1==0</tt> then returns zero (<tt>0L</tt>).
   *
   * @param from index of start bit (inclusive).
   * @param to   index of end bit (inclusive).
   * @return the specified bits as long value.
   * @throws IndexOutOfBoundsException if <tt>from&lt;0 || from&gt;=size() || to&lt;0 || to&gt;=size() || to-from+1<0 ||
   *                                   to-from+1>64</tt>
   */
  public long getLongFromTo(int from, int to) {
    int width = to - from + 1;
    if (width == 0) {
      return 0L;
    }
    if (from < 0 || from >= nbits || to < 0 || to >= nbits || width < 0 || width > QuickBitVector.BITS_PER_UNIT) {
      throw new IndexOutOfBoundsException("from:" + from + ", to:" + to);
    }
    return QuickBitVector.getLongFromTo(bits, from, to);
  }

  /**
   * Returns from the bitvector the value of the bit with the specified index; <b>WARNING:</b> Does not check
   * preconditions. The value is <tt>true</tt> if the bit with the index <tt>bitIndex</tt> is currently set; otherwise,
   * returns <tt>false</tt>.
   *
   * <p>Provided with invalid parameters this method may return invalid values without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the index is within bounds.</b> Precondition
   * (unchecked): <tt>bitIndex &gt;= 0 && bitIndex &lt; size()</tt>.
   *
   * @param bitIndex the bit index.
   * @return the value of the bit with the specified index.
   */
  public boolean getQuick(int bitIndex) {
    return QuickBitVector.get(bits, bitIndex);
  }

  /**
   * Returns a hash code value for the receiver. The hash code depends only on which bits have been set within the
   * receiver. The algorithm used to compute it may be described as follows.<p> Suppose the bits in the receiver were to
   * be stored in an array of <code>long</code> integers called, say, <code>bits</code>, in such a manner that bit
   * <code>k</code> is set in the receiver (for nonnegative values of <code>k</code>) if and only if the expression
   * <pre>((k&gt;&gt;6) &lt; bits.length) && ((bits[k&gt;&gt;6] & (1L &lt;&lt; (bit & 0x3F))) != 0)</pre>
   * is true. Then the following definition of the <code>hashCode</code> method would be a correct implementation of the
   * actual algorithm:
   * <pre>
   * public int hashCode() {
   *      long h = 1234;
   *      for (int i = bits.length; --i &gt;= 0; ) {
   *           h ^= bits[i] * (i + 1);
   *      }
   *      return (int)((h &gt;&gt; 32) ^ h);
   * }</pre>
   * Note that the hash code values change if the set of bits is altered.
   *
   * @return a hash code value for the receiver.
   */
  public int hashCode() {
    long h = 1234;
    for (int i = bits.length; --i >= 0;) {
      h ^= bits[i] * (i + 1);
    }

    return (int) ((h >> 32) ^ h);
  }

  /**
   * Returns the index of the first occurrence of the specified state. Returns <code>-1</code> if the receiver does not
   * contain this state. Searches between <code>from</code>, inclusive and <code>to</code>, inclusive. <p> Optimized for
   * speed. Preliminary performance (200Mhz Pentium Pro, JDK 1.2, NT): size=10^6, from=0, to=size-1, receiver contains
   * matching state in the very end --> 0.002 seconds elapsed time.
   *
   * @param state state to search for.
   * @param from  the leftmost search position, inclusive.
   * @param to    the rightmost search position, inclusive.
   * @return the index of the first occurrence of the element in the receiver; returns <code>-1</code> if the element is
   *         not found.
   * @throws IndexOutOfBoundsException if (<tt>size()&gt;0 && (from&lt;0 || from&gt;to || to&gt;=size())</tt>).
   */
  public int indexOfFromTo(int from, int to, boolean state) {
    IndexProcedure indexProcedure = new IndexProcedure();
    forEachIndexFromToInState(from, to, state, indexProcedure);
    return indexProcedure.foundPos;
  }

  /** Performs a logical <b>NOT</b> on the bits of the receiver (A = ~A). */
  public void not() {
    long[] theBits = this.bits;
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] = ~theBits[i];
    }
  }

  /**
   * Returns the number of bits used in the trailing PARTIAL unit. Returns zero if there is no such trailing partial
   * unit.
   */
  protected int numberOfBitsInPartialUnit() {
    return QuickBitVector.offset(nbits);
  }

  /** Returns the number of units that are FULL (not PARTIAL). */
  protected int numberOfFullUnits() {
    return QuickBitVector.unit(nbits);
  }

  /**
   * Performs a logical <b>OR</b> of the receiver with another bit vector (A = A | B). The receiver is modified so that
   * a bit in it has the value <code>true</code> if and only if it either already had the value <code>true</code> or the
   * corresponding bit in the other bit vector argument has the value <code>true</code>.
   *
   * @param other a bit vector.
   * @throws IllegalArgumentException if <tt>size() &gt; other.size()</tt>.
   */
  public void or(BitVector other) {
    if (this == other) {
      return;
    }
    checkSize(other);
    long[] theBits = this.bits; // cached for speed.
    long[] otherBits = other.bits; //cached for speed.
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] |= otherBits[i];
    }
  }

  /**
   * Constructs and returns a new bit vector which is a copy of the given range. The new bitvector has
   * <tt>size()==to-from+1</tt>.
   *
   * @param from the start index within the receiver, inclusive.
   * @param to   the end index within the receiver, inclusive.
   * @throws IndexOutOfBoundsException if <tt>size()&gt;0 && (from&lt;0 || from&gt;to || to&gt;=size()))</tt>.
   */
  public BitVector partFromTo(int from, int to) {
    if (nbits == 0 || to == from - 1) {
      return new BitVector(0);
    }
    checkRangeFromTo(from, to, nbits);

    int width = to - from + 1;
    BitVector part = new BitVector(width);
    part.replaceFromToWith(0, width - 1, this, from);
    return part;
  }

  /**
   * Sets the bit with index <tt>bitIndex</tt> to the state specified by <tt>value</tt>.
   *
   * @param bitIndex the index of the bit to be changed.
   * @param value    the value to be stored in the bit.
   * @throws IndexOutOfBoundsException if <tt>bitIndex&lt;0 || bitIndex&gt;=size()</tt>
   */
  public void put(int bitIndex, boolean value) {
    if (bitIndex < 0 || bitIndex >= nbits) {
      throw new IndexOutOfBoundsException(String.valueOf(bitIndex));
    }
    if (value) {
      QuickBitVector.set(bits, bitIndex);
    } else {
      QuickBitVector.clear(bits, bitIndex);
    }
  }

  /**
   * Sets bits of the receiver from index <code>from</code> to index <code>to</code> to the bits of <code>value</code>.
   * Bit <code>from</code> is set to bit 0 of <code>value</code>, ..., bit <code>to</code> is set to bit
   * <code>to-from</code> of <code>value</code>. All other bits stay unaffected. If <tt>to-from+1==0</tt> then does
   * nothing.
   *
   * @param value the value to be copied into the receiver.
   * @param from  index of start bit (inclusive).
   * @param to    index of end bit (inclusive).
   * @throws IndexOutOfBoundsException if <tt>from&lt;0 || from&gt;=size() || to&lt;0 || to&gt;=size() || to-from+1<0 ||
   *                                   to-from+1>64</tt>.
   */
  public void putLongFromTo(long value, int from, int to) {
    int width = to - from + 1;
    if (width == 0) {
      return;
    }
    if (from < 0 || from >= nbits || to < 0 || to >= nbits || width < 0 || width > QuickBitVector.BITS_PER_UNIT) {
      throw new IndexOutOfBoundsException("from:" + from + ", to:" + to);
    }
    QuickBitVector.putLongFromTo(bits, value, from, to);
  }

  /**
   * Sets the bit with index <tt>bitIndex</tt> to the state specified by <tt>value</tt>; <b>WARNING:</b> Does not check
   * preconditions.
   *
   * <p>Provided with invalid parameters this method may set invalid values without throwing any exception. <b>You
   * should only use this method when you are absolutely sure that the index is within bounds.</b> Precondition
   * (unchecked): <tt>bitIndex &gt;= 0 && bitIndex &lt; size()</tt>.
   *
   * @param bitIndex the index of the bit to be changed.
   * @param value    the value to be stored in the bit.
   */
  public void putQuick(int bitIndex, boolean value) {
    if (value) {
      QuickBitVector.set(bits, bitIndex);
    } else {
      QuickBitVector.clear(bits, bitIndex);
    }
  }

  /**
   * Replaces the bits of the receiver in the given range with the bits of another bit vector. Replaces the range
   * <tt>[from,to]</tt> with the contents of the range <tt>[sourceFrom,sourceFrom+to-from]</tt>, all inclusive. If
   * <tt>source==this</tt> and the source and destination range intersect in an ambiguous way, then replaces as if using
   * an intermediate auxiliary copy of the receiver. <p> Optimized for speed. Preliminary performance (200Mhz Pentium
   * Pro, JDK 1.2, NT): replace 10^6 ill aligned bits --> 0.02 seconds elapsed time.
   *
   * @param from       the start index within the receiver, inclusive.
   * @param to         the end index within the receiver, inclusive.
   * @param source     the source bitvector to copy from.
   * @param sourceFrom the start index within <tt>source</tt>, inclusive.
   * @throws IndexOutOfBoundsException if <tt>size()&gt;0 && (from&lt;0 || from&gt;to || to&gt;=size() || sourceFrom<0
   *                                   || sourceFrom+to-from+1>source.size()))</tt>.
   */
  public void replaceFromToWith(int from, int to, BitVector source, int sourceFrom) {
    if (nbits == 0 || to == from - 1) {
      return;
    }
    checkRangeFromTo(from, to, nbits);
    int length = to - from + 1;
    if (sourceFrom < 0 || sourceFrom + length > source.size()) {
      throw new IndexOutOfBoundsException();
    }

    if (source.bits == this.bits && from <= sourceFrom && sourceFrom <= to) { // dangerous intersection
      source = source.copy();
    }

    long[] theBits = this.bits; // cached for speed.
    long[] sourceBits = source.bits; // cached for speed.

    /*
    This version is equivalent to the version below but 20 times slower...
    for (int i=from; --length >= 0; i++, sourceFrom++) {
      QuickBitVector.put(theBits,i,QuickBitVector.get(sourceBits,sourceFrom));
    }
    */

    // Low level implementation for speed.
    // This could be done even faster by implementing on even lower levels. But then the code would probably become a "don't touch" piece.
    int width = to - from + 1;
    int blocks = QuickBitVector.unit(width); // width/64
    int bitsPerUnit = QuickBitVector.BITS_PER_UNIT;
    int bitsPerUnitMinusOne = bitsPerUnit - 1;

    // copy entire 64 bit blocks, if any.
    for (int i = blocks; --i >= 0;) {
      long val = QuickBitVector.getLongFromTo(sourceBits, sourceFrom, sourceFrom + bitsPerUnitMinusOne);
      QuickBitVector.putLongFromTo(theBits, val, from, from + bitsPerUnitMinusOne);
      sourceFrom += bitsPerUnit;
      from += bitsPerUnit;
    }

    // copy trailing bits, if any.
    int offset = QuickBitVector.offset(width); //width%64
    long val = QuickBitVector.getLongFromTo(sourceBits, sourceFrom, sourceFrom + offset - 1);
    QuickBitVector.putLongFromTo(theBits, val, from, from + offset - 1);
  }

  /**
   * Sets the bits in the given range to the state specified by <tt>value</tt>. <p> Optimized for speed. Preliminary
   * performance (200Mhz Pentium Pro, JDK 1.2, NT): replace 10^6 ill aligned bits --> 0.002 seconds elapsed time.
   *
   * @param from  the start index, inclusive.
   * @param to    the end index, inclusive.
   * @param value the value to be stored in the bits of the range.
   * @throws IndexOutOfBoundsException if <tt>size()&gt;0 && (from&lt;0 || from&gt;to || to&gt;=size())</tt>.
   */
  public void replaceFromToWith(int from, int to, boolean value) {
    if (nbits == 0 || to == from - 1) {
      return;
    }
    checkRangeFromTo(from, to, nbits);
    long[] theBits = this.bits; // cached for speed

    int fromUnit = QuickBitVector.unit(from);
    int fromOffset = QuickBitVector.offset(from);
    int toUnit = QuickBitVector.unit(to);
    int toOffset = QuickBitVector.offset(to);

    long filler;
    if (value) {
      filler = ~0L;
    } else {
      filler = 0L;
    }

    int bitIndex = from;
    if (fromUnit == toUnit) { // only one unit to do
      QuickBitVector.putLongFromTo(theBits, filler, bitIndex, bitIndex + to - from);
      //slower: for (; bitIndex<=to; ) QuickBitVector.put(theBits,bitIndex++,value);
      return;
    }

    // treat leading partial unit, if any.
    int bitsPerUnit = QuickBitVector.BITS_PER_UNIT;
    if (fromOffset > 0) { // fix by Olivier Janssens
      QuickBitVector.putLongFromTo(theBits, filler, bitIndex, bitIndex + bitsPerUnit - fromOffset);
      bitIndex += bitsPerUnit - fromOffset + 1;
      /* slower:
      for (int i=bitsPerUnit-fromOffset; --i >= 0; ) {
        QuickBitVector.put(theBits,bitIndex++,value);
      }*/
      fromUnit++;
    }
    if (toOffset < bitsPerUnit - 1) {
      toUnit--;
    } // there is a trailing partial unit

    // treat full units, if any.
    for (int i = fromUnit; i <= toUnit;) {
      theBits[i++] = filler;
    }
    if (fromUnit <= toUnit) {
      bitIndex += (toUnit - fromUnit + 1) * bitsPerUnit;
    }

    // treat trailing partial unit, if any.
    if (toOffset < bitsPerUnit - 1) {
      QuickBitVector.putLongFromTo(theBits, filler, bitIndex, to);
      /* slower:
      for (int i=toOffset+1; --i >= 0; ) {
        QuickBitVector.put(theBits,bitIndex++,value);
      }*/
    }
  }

  /**
   * Changes the bit with index <tt>bitIndex</tt> to the "set" (<tt>true</tt>) state.
   *
   * @param bitIndex the index of the bit to be set.
   * @throws IndexOutOfBoundsException if <tt>bitIndex&lt;0 || bitIndex&gt;=size()</tt>
   */
  public void set(int bitIndex) {
    if (bitIndex < 0 || bitIndex >= nbits) {
      throw new IndexOutOfBoundsException(String.valueOf(bitIndex));
    }
    QuickBitVector.set(bits, bitIndex);
  }

  /**
   * Shrinks or expands the receiver so that it holds <tt>newSize</tt> bits. If the receiver is expanded, additional
   * <tt>false</tt> bits are added to the end. If the receiver is shrinked, all bits between the old size and the new
   * size are lost; their memory is subject to garbage collection. (This method introduces a new backing array of
   * elements. WARNING: if you have more than one BitVector or BitMatrix sharing identical backing elements, be sure you
   * know what you are doing.)
   *
   * @param newSize the number of bits the bit vector shall have.
   * @throws IllegalArgumentException if <tt>size &lt; 0</tt>.
   */
  public void setSize(int newSize) {
    if (newSize != size()) {
      BitVector newVector = new BitVector(newSize);
      newVector.replaceFromToWith(0, Math.min(size(), newSize) - 1, this, 0);
      elements(newVector.elements(), newSize);
    }
  }

  /** Returns the size of the receiver. */
  public int size() {
    return nbits;
  }

  /**
   * Returns a string representation of the receiver. For every index for which the receiver contains a bit in the "set"
   * (<tt>true</tt>) state, the decimal representation of that index is included in the result. Such indeces are listed
   * in order from lowest to highest, separated by ",&nbsp;" (a comma and a space) and surrounded by braces.
   *
   * @return a string representation of this bit vector.
   */
  public String toString() {
    StringBuilder buffer = new StringBuilder(nbits);
    buffer.append('{');

    String separator = "";
    for (int i = 0; i < nbits; i++) {
      if (get(i)) {
        buffer.append(separator);
        separator = ", ";
        buffer.append(i);
      }
    }

    buffer.append('}');
    return buffer.toString();
  }

  /**
   * Performs a logical <b>XOR</b> of the receiver with another bit vector (A = A ^ B). The receiver is modified so that
   * a bit in it has the value <code>true</code> if and only if one of the following statements holds: <ul> <li>The bit
   * initially has the value <code>true</code>, and the corresponding bit in the argument has the value
   * <code>false</code>. <li>The bit initially has the value <code>false</code>, and the corresponding bit in the
   * argument has the value <code>true</code>. </ul>
   *
   * @param other a bit vector.
   * @throws IllegalArgumentException if <tt>size() &gt; other.size()</tt>.
   */
  public void xor(BitVector other) {
    checkSize(other);
    long[] theBits = this.bits; // cached for speed.
    long[] otherBits = other.bits; //cached for speed.
    for (int i = theBits.length; --i >= 0;) {
      theBits[i] ^= otherBits[i];
    }
  }
}
