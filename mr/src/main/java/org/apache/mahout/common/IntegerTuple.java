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
import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.WritableComparable;

/**
 * An Ordered List of Integers which can be used in a Hadoop Map/Reduce Job
 * 
 * 
 */
public final class IntegerTuple implements WritableComparable<IntegerTuple> {
  
  private List<Integer> tuple = Lists.newArrayList();
  
  public IntegerTuple() { }
  
  public IntegerTuple(Integer firstEntry) {
    add(firstEntry);
  }
  
  public IntegerTuple(Iterable<Integer> entries) {
    for (Integer entry : entries) {
      add(entry);
    }
  }
  
  public IntegerTuple(Integer[] entries) {
    for (Integer entry : entries) {
      add(entry);
    }
  }
  
  /**
   * add an entry to the end of the list
   * 
   * @param entry
   * @return true if the items get added
   */
  public boolean add(Integer entry) {
    return tuple.add(entry);
  }
  
  /**
   * Fetches the string at the given location
   * 
   * @param index
   * @return String value at the given location in the tuple list
   */
  public Integer integerAt(int index) {
    return tuple.get(index);
  }
  
  /**
   * Replaces the string at the given index with the given newString
   * 
   * @param index
   * @param newInteger
   * @return The previous value at that location
   */
  public Integer replaceAt(int index, Integer newInteger) {
    return tuple.set(index, newInteger);
  }
  
  /**
   * Fetch the list of entries from the tuple
   * 
   * @return a List containing the strings in the order of insertion
   */
  public List<Integer> getEntries() {
    return Collections.unmodifiableList(this.tuple);
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
  public String toString() {
    return tuple.toString();
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
    IntegerTuple other = (IntegerTuple) obj;
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
    tuple = Lists.newArrayListWithCapacity(len);
    for (int i = 0; i < len; i++) {
      int data = in.readInt();
      tuple.add(data);
    }
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(tuple.size());
    for (Integer entry : tuple) {
      out.writeInt(entry);
    }
  }
  
  @Override
  public int compareTo(IntegerTuple otherTuple) {
    int thisLength = length();
    int otherLength = otherTuple.length();
    int min = Math.min(thisLength, otherLength);
    for (int i = 0; i < min; i++) {
      int ret = this.tuple.get(i).compareTo(otherTuple.integerAt(i));
      if (ret == 0) {
        continue;
      }
      return ret;
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
