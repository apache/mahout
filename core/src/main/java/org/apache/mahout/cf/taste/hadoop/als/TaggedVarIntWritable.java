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

public class TaggedVarIntWritable implements WritableComparable<TaggedVarIntWritable> {

  private int value;
  private boolean tagged;

  static {
    WritableComparator.define(TaggedVarIntWritable.class, new SecondarySortComparator());
  }

  public TaggedVarIntWritable() {
  }

  public TaggedVarIntWritable(int value, boolean tagged) {
    this.value = value;
    this.tagged = tagged;
  }

  public int get() {
    return value;
  }

  @Override
  public int compareTo(TaggedVarIntWritable other) {
    return value == other.value ? 0 : value < other.value ? -1 : 1;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(tagged);
    Varint.writeSignedVarInt(value, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    tagged = in.readBoolean();
    value = Varint.readSignedVarInt(in);
  }

  @Override
  public int hashCode() {
    return value;
  }


  @Override
  public boolean equals(Object o) {
    if (o instanceof TaggedVarIntWritable) {
      TaggedVarIntWritable other = (TaggedVarIntWritable) o;
      return value == other.value;
    }
    return false;
  }

  public static class SecondarySortComparator extends WritableComparator implements Serializable {

    protected SecondarySortComparator() {
      super(TaggedVarIntWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      TaggedVarIntWritable first = (TaggedVarIntWritable) a;
      TaggedVarIntWritable second = (TaggedVarIntWritable) b;

      int result = compare(first.value, second.value);
      if (result == 0) {
        if (first.tagged && !second.tagged) {
          return -1;
        } else if (!first.tagged && second.tagged) {
          return 1;
        }
      }
      return result;
    }

    protected static int compare(int a, int b) {
      return a == b ? 0 : a < b ? -1 : 1;
    }
  }

  public static class GroupingComparator extends WritableComparator implements Serializable {

    protected GroupingComparator() {
      super(TaggedVarIntWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      return a.compareTo(b);
    }
  }
}
