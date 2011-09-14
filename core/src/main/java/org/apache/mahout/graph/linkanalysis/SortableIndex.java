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

package org.apache.mahout.graph.linkanalysis;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

/* a "weighted" IntWritable that can be used for secondary sort */
public class SortableIndex implements WritableComparable<SortableIndex> {

  private int index;
  private double weight;

  static {
    WritableComparator.define(SortableIndex.class, new SecondarySortComparator());
  }

  public SortableIndex() {}

  public SortableIndex(int index, double weight) {
    this.index = index;
    this.weight = weight;
  }

  public int get() {
    return index;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(index);
    out.writeDouble(weight);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    index = in.readInt();
    weight = in.readDouble();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof SortableIndex) {
      return index == ((SortableIndex) o).index;
    }
    return false;
  }

  @Override
  public int hashCode() {
    return index;
  }

  @Override
  public int compareTo(SortableIndex other) {
    return Ints.compare(index, other.index);
  }

  public static class SecondarySortComparator extends WritableComparator implements Serializable {

    protected SecondarySortComparator() {
      super(SortableIndex.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      SortableIndex first = (SortableIndex) a;
      SortableIndex second = (SortableIndex) b;

      int result = first.compareTo(second);
      if (result == 0) {
        result = -1 * Doubles.compare(first.weight, second.weight);
      }
      return result;
    }
  }

  public static class GroupingComparator extends WritableComparator implements Serializable {

    protected GroupingComparator() {
      super(SortableIndex.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      return a.compareTo(b);
    }
  }

}