/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math.bitvector;

/**
 * Implements quick non polymorphic non bounds checking low level bitvector operations.
 * Includes some operations that interpret sub-bitstrings as long integers.
 * <p>
 * <b>WARNING: Methods of this class do not check preconditions.</b>
 * Provided with invalid parameters these method may return (or set) invalid values without throwing any exception.
 * <b>You should only use this class when performance is critical and you are absolutely sure that indexes are within bounds.</b>
 * <p>   
 * A bitvector is modelled as a long array, i.e. <tt>long[] bits</tt> holds bits of a bitvector.
 * Each long value holds 64 bits.
 * The i-th bit is stored in bits[i/64] at
 * bit position i % 64 (where bit position 0 refers to the least
 * significant bit and 63 refers to the most significant bit).
 *
 * @see     BitVector
 * @see     BitMatrix
 * @see     java.util.BitSet
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class QuickBitVector {

  private static final int ADDRESS_BITS_PER_UNIT = 6; // 64=2^6
  protected static final int BITS_PER_UNIT = 64; // = 1 << ADDRESS_BITS_PER_UNIT
  private static final int BIT_INDEX_MASK = 63; // = BITS_PER_UNIT - 1;

  private static final long[] pows = precomputePows(); //precompute bitmasks for speed

  /** Makes this class non instantiable, but still inheritable. */
  private QuickBitVector() {
  }

  /**
   * Returns a bit mask with bits in the specified range set to 1, all the rest set to 0. In other words, returns a bit
   * mask having 0,1,2,3,...,64 bits set. If <tt>to-from+1==0</tt> then returns zero (<tt>0L</tt>). Precondition (not
   * checked): <tt>to-from+1 &gt;= 0 && to-from+1 &lt;= 64</tt>.
   *
   * @param from index of start bit (inclusive)
   * @param to   index of end bit (inclusive).
   * @return the bit mask having all bits between <tt>from</tt> and <tt>to</tt> set to 1.
   */
  private static long bitMaskWithBitsSetFromTo(int from, int to) {
    return pows[to - from + 1] << from;

    // This turned out to be slower:
    // 0xffffffffffffffffL == ~0L == -1L == all 64 bits set.
    // int width;
    // return (width=to-from+1) == 0 ? 0L : (0xffffffffffffffffL >>> (BITS_PER_UNIT-width)) << from;
  }

  /**
   * Changes the bit with index <tt>bitIndex</tt> in the bitvector <tt>bits</tt> to the "clear" (<tt>false</tt>) state.
   *
   * @param bits     the bitvector.
   * @param bitIndex the index of the bit to be cleared.
   */
  public static void clear(long[] bits, int bitIndex) {
    bits[bitIndex >> ADDRESS_BITS_PER_UNIT] &= ~(1L << (bitIndex & BIT_INDEX_MASK));
  }

  /**
   * Returns from the bitvector the value of the bit with the specified index. The value is <tt>true</tt> if the bit
   * with the index <tt>bitIndex</tt> is currently set; otherwise, returns <tt>false</tt>.
   *
   * @param bits     the bitvector.
   * @param bitIndex the bit index.
   * @return the value of the bit with the specified index.
   */
  public static boolean get(long[] bits, int bitIndex) {
    return ((bits[bitIndex >> ADDRESS_BITS_PER_UNIT] & (1L << (bitIndex & BIT_INDEX_MASK))) != 0);
  }

  /**
   * Returns a long value representing bits of a bitvector from index <tt>from</tt> to index <tt>to</tt>. Bits are
   * returned as a long value with the return value having bit 0 set to bit <code>from</code>, ..., bit
   * <code>to-from</code> set to bit <code>to</code>. All other bits of return value are set to 0. If <tt>from &gt;
   * to</tt> then returns zero (<tt>0L</tt>). Precondition (not checked): <tt>to-from+1 &lt;= 64</tt>.
   *
   * @param bits the bitvector.
   * @param from index of start bit (inclusive).
   * @param to   index of end bit (inclusive).
   * @return the specified bits as long value.
   */
  public static long getLongFromTo(long[] bits, int from, int to) {
    if (from > to) {
      return 0L;
    }

    int fromIndex = from >> ADDRESS_BITS_PER_UNIT; //equivalent to from/64
    int toIndex = to >> ADDRESS_BITS_PER_UNIT;
    int fromOffset = from & BIT_INDEX_MASK; //equivalent to from%64
    int toOffset = to & BIT_INDEX_MASK;
    //this is equivalent to the above, but slower:
    //final int fromIndex=from/BITS_PER_UNIT;
    //final int toIndex=to/BITS_PER_UNIT;
    //final int fromOffset=from%BITS_PER_UNIT;
    //final int toOffset=to%BITS_PER_UNIT;


    long mask;
    if (fromIndex ==
        toIndex) { //range does not cross unit boundaries; value to retrieve is contained in one single long value.
      mask = bitMaskWithBitsSetFromTo(fromOffset, toOffset);
      return (bits[fromIndex] & mask) >>> fromOffset;

    }

    //range crosses unit boundaries; value to retrieve is spread over two long values.
    //get part from first long value
    mask = bitMaskWithBitsSetFromTo(fromOffset, BIT_INDEX_MASK);
    long x1 = (bits[fromIndex] & mask) >>> fromOffset;

    //get part from second long value
    mask = bitMaskWithBitsSetFromTo(0, toOffset);
    long x2 = (bits[toIndex] & mask) << (BITS_PER_UNIT - fromOffset);

    //combine
    return x1 | x2;
  }

  /**
   * Returns the index of the least significant bit in state "true". Returns 32 if no bit is in state "true". Examples:
   * <pre>
   * 0x80000000 --> 31
   * 0x7fffffff --> 0
   * 0x00000001 --> 0
   * 0x00000000 --> 32
   * </pre>
   */
  public static int leastSignificantBit(int value) {
    int i = -1;
    while (++i < 32 && (((1 << i) & value)) == 0) {
    }
    return i;
  }

  /**
   * Constructs a low level bitvector that holds <tt>size</tt> elements, with each element taking
   * <tt>bitsPerElement</tt> bits.
   *
   * @param size           the number of elements to be stored in the bitvector (must be &gt;= 0).
   * @param bitsPerElement the number of bits one single element takes.
   * @return a low level bitvector.
   */
  public static long[] makeBitVector(int size, int bitsPerElement) {
    int nBits = size * bitsPerElement;
    int unitIndex = (nBits - 1) >> ADDRESS_BITS_PER_UNIT;
    return new long[unitIndex + 1];
  }

  /**
   * Returns the index of the most significant bit in state "true". Returns -1 if no bit is in state "true". Examples:
   * <pre>
   * 0x80000000 --> 31
   * 0x7fffffff --> 30
   * 0x00000001 --> 0
   * 0x00000000 --> -1
   * </pre>
   */
  public static int mostSignificantBit(int value) {
    int i = 32;
    while (--i >= 0 && (((1 << i) & value)) == 0) {
    }
    return i;
  }

  /** Returns the index within the unit that contains the given bitIndex. */
  protected static int offset(int bitIndex) {
    return bitIndex & BIT_INDEX_MASK;
    //equivalent to bitIndex%64
  }

  /**
   * Initializes a table with numbers having 1,2,3,...,64 bits set. pows[i] has bits [0..i-1] set. pows[64] == -1L ==
   * ~0L == has all 64 bits set --> correct. to speedup calculations in subsequent methods.
   */
  private static long[] precomputePows() {
    long[] pows = new long[BITS_PER_UNIT + 1];
    long value = ~0L;
    for (int i = BITS_PER_UNIT + 1; --i >= 1;) {
      pows[i] = value >>> (BITS_PER_UNIT - i);
    }
    pows[0] = 0L;
    return pows;
  }

  /**
   * Sets the bit with index <tt>bitIndex</tt> in the bitvector <tt>bits</tt> to the state specified by <tt>value</tt>.
   *
   * @param bits     the bitvector.
   * @param bitIndex the index of the bit to be changed.
   * @param value    the value to be stored in the bit.
   */
  public static void put(long[] bits, int bitIndex, boolean value) {
    if (value) {
      set(bits, bitIndex);
    } else {
      clear(bits, bitIndex);
    }
  }

  /**
   * Sets bits of a bitvector from index <code>from</code> to index <code>to</code> to the bits of <code>value</code>.
   * Bit <code>from</code> is set to bit 0 of <code>value</code>, ..., bit <code>to</code> is set to bit
   * <code>to-from</code> of <code>value</code>. All other bits stay unaffected. If <tt>from &gt; to</tt> then does
   * nothing. Precondition (not checked): <tt>to-from+1 &lt;= 64</tt>.
   *
   * @param bits  the bitvector.
   * @param value the value to be copied into the bitvector.
   * @param from  index of start bit (inclusive).
   * @param to    index of end bit (inclusive).
   */
  public static void putLongFromTo(long[] bits, long value, int from, int to) {
    if (from > to) {
      return;
    }

    int fromIndex = from >> ADDRESS_BITS_PER_UNIT; //equivalent to from/64
    int toIndex = to >> ADDRESS_BITS_PER_UNIT;
    int fromOffset = from & BIT_INDEX_MASK; //equivalent to from%64
    int toOffset = to & BIT_INDEX_MASK;
    /*
    this is equivalent to the above, but slower:
    int fromIndex=from/BITS_PER_UNIT;
    int toIndex=to/BITS_PER_UNIT;
    int fromOffset=from%BITS_PER_UNIT;
    int toOffset=to%BITS_PER_UNIT;
    */

    //make sure all unused bits to the left are cleared.
    long mask = bitMaskWithBitsSetFromTo(to - from + 1, BIT_INDEX_MASK);
    long cleanValue = value & (~mask);

    long shiftedValue;

    if (fromIndex == toIndex) { //range does not cross unit boundaries; should go into one single long value.
      shiftedValue = cleanValue << fromOffset;
      mask = bitMaskWithBitsSetFromTo(fromOffset, toOffset);
      bits[fromIndex] = (bits[fromIndex] & (~mask)) | shiftedValue;
      return;

    }

    //range crosses unit boundaries; value should go into two long values.
    //copy into first long value.
    shiftedValue = cleanValue << fromOffset;
    mask = bitMaskWithBitsSetFromTo(fromOffset, BIT_INDEX_MASK);
    bits[fromIndex] = (bits[fromIndex] & (~mask)) | shiftedValue;

    //copy into second long value.
    shiftedValue = cleanValue >>> (BITS_PER_UNIT - fromOffset);
    mask = bitMaskWithBitsSetFromTo(0, toOffset);
    bits[toIndex] = (bits[toIndex] & (~mask)) | shiftedValue;
  }

  /**
   * Changes the bit with index <tt>bitIndex</tt> in the bitvector <tt>bits</tt> to the "set" (<tt>true</tt>) state.
   *
   * @param bits     the bitvector.
   * @param bitIndex the index of the bit to be set.
   */
  public static void set(long[] bits, int bitIndex) {
    bits[bitIndex >> ADDRESS_BITS_PER_UNIT] |= 1L << (bitIndex & BIT_INDEX_MASK);
  }

  /** Returns the index of the unit that contains the given bitIndex. */
  protected static int unit(int bitIndex) {
    return bitIndex >> ADDRESS_BITS_PER_UNIT;
    //equivalent to bitIndex/64
  }
}
