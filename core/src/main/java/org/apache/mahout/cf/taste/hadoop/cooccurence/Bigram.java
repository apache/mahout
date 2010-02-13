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

package org.apache.mahout.cf.taste.hadoop.cooccurence;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableUtils;

public final class Bigram implements WritableComparable<Bigram> {
  
  private int first;
  private int second;
  
  public Bigram() {
    set(-1, -1);
  }
  
  public Bigram(Bigram bigram) {
    set(bigram.first, bigram.second);
  }
  
  public Bigram(int first, int second) {
    set(first, second);
  }
  
  public void set(int first, int second) {
    this.first = first;
    this.second = second;
  }
  
  public int getFirst() {
    return first;
  }
  
  public int getSecond() {
    return second;
  }
  
  /** Read the two integers encoded using variable length encoding */
  @Override
  public void readFields(DataInput in) throws IOException {
    first = WritableUtils.readVInt(in);
    second = WritableUtils.readVInt(in);
  }
  
  /** Write the two integers encoded using variable length encoding */
  @Override
  public void write(DataOutput out) throws IOException {
    WritableUtils.writeVInt(out, first);
    WritableUtils.writeVInt(out, second);
  }
  
  @Override
  public int hashCode() {
    return first * 157 + second;
  }
  
  @Override
  public boolean equals(Object right) {
    if (right == null) {
      return false;
    }
    if (right instanceof Bigram) {
      Bigram r = (Bigram) right;
      return (r.getFirst() == first) && (r.second == second);
    } else {
      return false;
    }
  }
  
  @Override
  public int compareTo(Bigram o) {
    if (first == o.first) {
      if (second == o.second) {
        return 0;
      } else {
        return second < o.second ? -1 : 1;
      }
    } else {
      return first < o.first ? -1 : 1;
    }
  }
  
  @Override
  public String toString() {
    return first + "\t" + second;
  }
  
  /** A Comparator that compares serialized Bigrams. */
  public static class Comparator extends WritableComparator implements Serializable {
    
    public Comparator() {
      super(Bigram.class);
    }
    
    /** Compare varibale length encoded numbers * */
    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      int ret;
      try {
        int firstb1 = WritableComparator.readVInt(b1, s1);
        int firstb2 = WritableComparator.readVInt(b2, s2);
        ret = firstb1 - firstb2;
        if (ret == 0) {
          int secondb1 = WritableComparator.readVInt(b1, s1 + WritableUtils.decodeVIntSize(b1[s1]));
          int secondb2 = WritableComparator.readVInt(b2, s2 + WritableUtils.decodeVIntSize(b2[s2]));
          ret = secondb1 - secondb2;
        }
      } catch (IOException ioe) {
        throw new IllegalArgumentException(ioe);
      }
      return ret;
    }
  }
  
  static { // register this comparator
    WritableComparator.define(Bigram.class, new Comparator());
  }
  
  /**
   * Compare only the first part of the bigram, so that reduce is called once for each value of the first
   * part.
   */
  public static class FirstGroupingComparator extends WritableComparator implements Serializable {
    
    public FirstGroupingComparator() {
      super(Bigram.class);
    }
    
    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      int ret;
      try {
        int firstb1 = WritableComparator.readVInt(b1, s1);
        int firstb2 = WritableComparator.readVInt(b2, s2);
        ret = firstb1 - firstb2;
      } catch (IOException ioe) {
        throw new IllegalArgumentException(ioe);
      }
      return ret;
    }
    
    @Override
    public int compare(Object o1, Object o2) {
      if (o1 == null) {
        return -1;
      } else if (o2 == null) {
        return 1;
      } else {
        int firstb1 = ((Bigram) o1).getFirst();
        int firstb2 = ((Bigram) o2).getFirst();
        return firstb1 - firstb2;
      }
    }
    
  }
  
  /** A wrapper class that associates pairs with frequency (Occurences) */
  public static class Frequency implements Comparable<Frequency> {
    
    private Bigram bigram = new Bigram();
    private double frequency = 0.0;
    
    public double getFrequency() {
      return frequency;
    }
    
    public Bigram getBigram() {
      return bigram;
    }
    
    public Frequency(Bigram bigram, double frequency) {
      this.bigram = new Bigram(bigram);
      this.frequency = frequency;
    }
    
    @Override
    public int hashCode() {
      return bigram.hashCode() + (int) Math.abs(Math.round(frequency * 31));
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
      return bigram + "\t" + frequency;
    }
  }
}