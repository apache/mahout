/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.common;

import org.apache.hadoop.io.BinaryComparable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;

/**
 * A {@link WritableComparable} which encapsulates an ordered pair of signed integers.
 */
public final class IntPairWritable extends BinaryComparable
    implements WritableComparable<BinaryComparable>, Cloneable {

  static final int INT_BYTE_LENGTH = 4;
  static final int INT_PAIR_BYTE_LENGTH = 2 * INT_BYTE_LENGTH;
  private byte[] b = new byte[INT_PAIR_BYTE_LENGTH];
  
  public IntPairWritable() {
    setFirst(0);
    setSecond(0);
  }
  
  public IntPairWritable(IntPairWritable pair) {
    b = Arrays.copyOf(pair.getBytes(), INT_PAIR_BYTE_LENGTH);
  }
  
  public IntPairWritable(int x, int y) {
    putInt(x, b, 0);
    putInt(y, b, INT_BYTE_LENGTH);
  }
  
  public void set(int x, int y) {
    putInt(x, b, 0);
    putInt(y, b, INT_BYTE_LENGTH);
  }
  
  public void setFirst(int x) {
    putInt(x, b, 0);
  }
  
  public int getFirst() {
    return getInt(b, 0);
  }
  
  public void setSecond(int y) {
    putInt(y, b, INT_BYTE_LENGTH);
  }
  
  public int getSecond() {
    return getInt(b, INT_BYTE_LENGTH);
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    in.readFully(b);
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.write(b);
  }
  
  @Override
  public int hashCode() {
    return Arrays.hashCode(b);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!super.equals(obj)) {
      return false;
    }
    if (!(obj instanceof IntPairWritable)) {
      return false;
    }
    IntPairWritable other = (IntPairWritable) obj;
    return Arrays.equals(b, other.b);
  }

  @Override
  public int compareTo(BinaryComparable other) {
    return Comparator.doCompare(b, 0, ((IntPairWritable) other).b, 0);
  }

  @Override
  public Object clone() {
    return new IntPairWritable(this);
  }
  
  @Override
  public String toString() {
    return "(" + getFirst() + ", " + getSecond() + ')';
  }
  
  @Override
  public byte[] getBytes() {
    return b;
  }
  
  @Override
  public int getLength() {
    return INT_PAIR_BYTE_LENGTH;
  }
  
  private static void putInt(int value, byte[] b, int offset) {
    for (int i = offset, j = 24; j >= 0; i++, j -= 8) {
      b[i] = (byte) (value >> j);
    }
  }
  
  private static int getInt(byte[] b, int offset) {
    int value = 0;
    for (int i = offset, j = 24; j >= 0; i++, j -= 8) {
      value |= (b[i] & 0xFF) << j;
    }
    return value;
  }

  static {
    WritableComparator.define(IntPairWritable.class, new Comparator());
  }

  public static final class Comparator extends WritableComparator implements Serializable {
    public Comparator() {
      super(IntPairWritable.class);
    }
    
    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      return doCompare(b1, s1, b2, s2);
    }

    static int doCompare(byte[] b1, int s1, byte[] b2, int s2) {
      int compare1 = compareInts(b1, s1, b2, s2);
      if (compare1 != 0) {
        return compare1;
      }
      return compareInts(b1, s1 + INT_BYTE_LENGTH, b2, s2 + INT_BYTE_LENGTH);
    }

    private static int compareInts(byte[] b1, int s1, byte[] b2, int s2) {
      // Like WritableComparator.compareBytes(), but treats first byte as signed value
      int end1 = s1 + INT_BYTE_LENGTH;
      for (int i = s1, j = s2; i < end1; i++, j++) {
        int a = b1[i];
        int b = b2[j];
        if (i > s1) {
          a &= 0xff;
          b &= 0xff;
        }
        if (a != b) {
          return a - b;
        }
      }
      return 0;
    }
  }
  
  /**
   * Compare only the first part of the pair, so that reduce is called once for each value of the first part.
   */
  public static class FirstGroupingComparator extends WritableComparator implements Serializable {
    
    public FirstGroupingComparator() {
      super(IntPairWritable.class);
    }
    
    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      int firstb1 = WritableComparator.readInt(b1, s1);
      int firstb2 = WritableComparator.readInt(b2, s2);
      if (firstb1 < firstb2) {
        return -1;
      } else if (firstb1 > firstb2) {
        return 1;
      } else {
        return 0;
      }
    }
    
    @Override
    public int compare(Object o1, Object o2) {
      int firstb1 = ((IntPairWritable) o1).getFirst();
      int firstb2 = ((IntPairWritable) o2).getFirst();
      if (firstb1 < firstb2) {
        return -1;
      }
      if (firstb1 > firstb2) {
        return 1;
      }
      return 0;
    }
    
  }
  
  /** A wrapper class that associates pairs with frequency (Occurrences) */
  public static class Frequency implements Comparable<Frequency>, Serializable {
    
    private final IntPairWritable pair;
    private final double frequency;

    public Frequency(IntPairWritable bigram, double frequency) {
      this.pair = new IntPairWritable(bigram);
      this.frequency = frequency;
    }

    public double getFrequency() {
      return frequency;
    }

    public IntPairWritable getPair() {
      return pair;
    }

    @Override
    public int hashCode() {
      return pair.hashCode() + RandomUtils.hashDouble(frequency);
    }
    
    @Override
    public boolean equals(Object right) {
      if (!(right instanceof Frequency)) {
        return false;
      }
      Frequency that = (Frequency) right;
      return pair.equals(that.pair) && frequency == that.frequency;
    }
    
    @Override
    public int compareTo(Frequency that) {
      if (frequency < that.frequency) {
        return -1;
      }
      if (frequency > that.frequency) {
        return 1;
      }
      return 0;
    }
    
    @Override
    public String toString() {
      return pair + "\t" + frequency;
    }
  }
}
