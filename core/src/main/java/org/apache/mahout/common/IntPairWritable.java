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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;

import org.apache.hadoop.io.BinaryComparable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
 * Saves two ints, x and y.
 */
public final class IntPairWritable extends BinaryComparable implements WritableComparable<BinaryComparable> {
  
  private static final int INT_PAIR_BYTE_LENGTH = 8;
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
    putInt(y, b, 4);
  }
  
  public void set(int x, int y) {
    putInt(x, b, 0);
    putInt(y, b, 4);
  }
  
  public void setFirst(int x) {
    putInt(x, b, 0);
  }
  
  public int getFirst() {
    return getInt(b, 0);
  }
  
  public void setSecond(int y) {
    putInt(y, b, 4);
  }
  
  public int getSecond() {
    return getInt(b, 4);
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
    return 43 * Arrays.hashCode(b);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (!super.equals(obj)) return false;
    if (getClass() != obj.getClass()) return false;
    IntPairWritable other = (IntPairWritable) obj;
    if (!Arrays.equals(b, other.b)) return false;
    return true;
  }
  
  @Override
  public String toString() {
    return "(" + getFirst() + ", " + getSecond() + ")";
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
    if (offset + 4 > INT_PAIR_BYTE_LENGTH) {
      throw new IllegalArgumentException("offset+4 exceeds byte array length");
    }
    
    for (int i = 0; i < 4; i++) {
      b[offset + i] = (byte) (((value >>> ((3 - i) * 8)) & 0xFF) ^ 0x80);
    }
  }
  
  private static int getInt(byte[] b, int offset) {
    if (offset + 4 > INT_PAIR_BYTE_LENGTH) {
      throw new IllegalArgumentException("offset+4 exceeds byte array length");
    }
    
    int value = 0;
    for (int i = 0; i < 4; i++) {
      value += ((b[i + offset] & 0xFF) ^ 0x80) << (3 - i) * 8;
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
      if (l1 != 8 || l2 != 8) {
        throw new IllegalArgumentException();
      }
      return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
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
      int ret;
      int firstb1 = WritableComparator.readInt(b1, s1);
      int firstb2 = WritableComparator.readInt(b2, s2);
      ret = firstb1 - firstb2;
      return ret;
    }
    
    @Override
    public int compare(Object o1, Object o2) {
      if (o1 == null) {
        return -1;
      } else if (o2 == null) {
        return 1;
      } else {
        int firstb1 = ((IntPairWritable) o1).getFirst();
        int firstb2 = ((IntPairWritable) o2).getFirst();
        return firstb1 - firstb2;
      }
    }
    
  }
  
  /** A wrapper class that associates pairs with frequency (Occurences) */
  public static class Frequency implements Comparable<Frequency> {
    
    private IntPairWritable pair = new IntPairWritable();
    private double frequency = 0.0;
    
    public double getFrequency() {
      return frequency;
    }
    
    public IntPairWritable getPair() {
      return pair;
    }
    
    public Frequency(IntPairWritable bigram, double frequency) {
      this.pair = new IntPairWritable(bigram);
      this.frequency = frequency;
    }
    
    @Override
    public int hashCode() {
      return pair.hashCode() + (int) Math.abs(Math.round(frequency * 31));
    }
    
    @Override
    public boolean equals(Object right) {
      if ((right == null) || !(right instanceof Frequency)) {
        return false;
      }
      Frequency that = (Frequency) right;
      return this.compareTo(that) == 0;
    }
    
    @Override
    public int compareTo(Frequency that) {
      return this.frequency > that.frequency ? 1 : -1;
    }
    
    @Override
    public String toString() {
      return pair + "\t" + frequency;
    }
  }
}
