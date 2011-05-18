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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.mahout.math.Varint;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

public class IndexedVarIntWritable implements WritableComparable<IndexedVarIntWritable> {

  private int value;
  private int index;

  static {
    WritableComparator.define(IndexedVarIntWritable.class, new SecondarySortComparator());
  }

  public IndexedVarIntWritable() {
  }

  public IndexedVarIntWritable(int value, int index) {
    this.value = value;
    this.index = index;
  }

  public int getValue() {
    return value;
  }

  @Override
  public int compareTo(IndexedVarIntWritable other) {
    return value == other.value ? 0 : value < other.value ? -1 : 1;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarInt(value, out);
    Varint.writeSignedVarInt(index, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    value = Varint.readSignedVarInt(in);
    index = Varint.readSignedVarInt(in);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof IndexedVarIntWritable) {
      return value == ((IndexedVarIntWritable) o).value;
    }
    return false;
  }

  @Override
  public int hashCode() {
    return value;
  }

  public static class SecondarySortComparator extends WritableComparator implements Serializable {

    protected SecondarySortComparator() {
      super(IndexedVarIntWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      IndexedVarIntWritable first = (IndexedVarIntWritable) a;
      IndexedVarIntWritable second = (IndexedVarIntWritable) b;

      int result = compare(first.value, second.value);
      if (result == 0) {
        result = compare(first.index, second.index);
      }
      return result;
    }

    protected static int compare(int a, int b) {
      return a == b ? 0 : a < b ? -1 : 1;
    }
  }

  public static class GroupingComparator extends WritableComparator implements Serializable {

    protected GroupingComparator() {
      super(IndexedVarIntWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      return a.compareTo(b);
    }
  }

}
