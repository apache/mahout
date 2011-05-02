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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.list.IntArrayList;

/**
 * An Ordered List of Integers which can be used in a Hadoop Map/Reduce Job
 */
public final class IntTuple implements WritableComparable<IntTuple> {
  
  private IntArrayList tuple = new IntArrayList();
  
  public IntTuple() {
  }
  
  public IntTuple(int firstEntry) {
    add(firstEntry);
  }
  
  public IntTuple(Iterable<Integer> entries) {
    for (Integer entry : entries) {
      add(entry);
    }
  }
  
  public IntTuple(int[] entries) {
    for (int entry : entries) {
      add(entry);
    }
  }
  
  /**
   * Add an entry to the end of the list
   */
  public void add(int entry) {
    tuple.add(entry);
  }
  
  /**
   * Fetches the string at the given location
   * 
   * @return int value at the given location in the tuple list
   */
  public int at(int index) {
    return tuple.get(index);
  }
  
  /**
   * Replaces the string at the given index with the given newInteger
   * 
   * @return The previous value at that location
   */
  public int replaceAt(int index, int newInteger) {
    int old = tuple.get(index);
    tuple.set(index, newInteger);
    return old;
  }
  
  /**
   * Fetch the list of entries from the tuple
   * 
   * @return a List containing the strings in the order of insertion
   */
  public IntArrayList getEntries() {
    return new IntArrayList(this.tuple.elements());
  }
  
  /**
   * Returns the length of the tuple
   * 
   * @return length
   */
  public int length() {
    return this.tuple.size();
  }
  
  @Override
  public int hashCode() {
    return tuple.hashCode();
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    IntTuple other = (IntTuple) obj;
    if (tuple == null) {
      if (other.tuple != null) {
        return false;
      }
    } else if (!tuple.equals(other.tuple)) {
      return false;
    }
    return true;
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int len = in.readInt();
    tuple = new IntArrayList(len);
    IntWritable value = new IntWritable();
    for (int i = 0; i < len; i++) {
      value.readFields(in);
      tuple.add(value.get());
    }
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(tuple.size());
    IntWritable value = new IntWritable();
    for (int entry : tuple.elements()) {
      value.set(entry);
      value.write(out);
    }
  }
  
  @Override
  public int compareTo(IntTuple otherTuple) {
    int thisLength = length();
    int otherLength = otherTuple.length();
    int min = Math.min(thisLength, otherLength);
    for (int i = 0; i < min; i++) {
      int a = this.tuple.get(i);
      int b = otherTuple.at(i);
      if (a < b) {
        return -1;
      } else if (a > b) {
        return 1;
      }
    }
    if (thisLength < otherLength) {
      return -1;
    } else if (thisLength > otherLength) {
      return 1;
    } else {
      return 0;
    }
  }
  
}
