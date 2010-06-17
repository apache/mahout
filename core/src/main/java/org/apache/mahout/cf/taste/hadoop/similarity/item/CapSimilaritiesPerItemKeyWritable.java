/*
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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Varint;

/**
 * used as key for the {@link CapSimilaritiesPerItemReducer} to collect all items similar to the item with the itemID
 *
 * ensure that the similar items are seen in descending order by their similarity value via secondary sort
 */
public class CapSimilaritiesPerItemKeyWritable implements WritableComparable<CapSimilaritiesPerItemKeyWritable> {

  private long itemID;
  private double associatedSimilarity;

  static {
    WritableComparator.define(CapSimilaritiesPerItemKeyWritable.class, new CapSimilaritiesPerItemKeyComparator());
  }

  public CapSimilaritiesPerItemKeyWritable() {
    super();
  }

  public CapSimilaritiesPerItemKeyWritable(long itemID, double associatedSimilarity) {
    super();
    this.itemID = itemID;
    this.associatedSimilarity = associatedSimilarity;
  }

  public long getItemID() {
    return itemID;
  }

  public double getAssociatedSimilarity() {
    return associatedSimilarity;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemID = Varint.readSignedVarLong(in);
    associatedSimilarity = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarLong(itemID, out);
    out.writeDouble(associatedSimilarity);
  }

  @Override
  public int compareTo(CapSimilaritiesPerItemKeyWritable other) {
    return (itemID == other.itemID) ? 0 : (itemID < other.itemID) ? -1 : 1;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(itemID);
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof CapSimilaritiesPerItemKeyWritable) {
      return itemID == ((CapSimilaritiesPerItemKeyWritable)other).itemID;
    }
    return false;
  }

  public static class CapSimilaritiesPerItemKeyComparator extends WritableComparator {

    public CapSimilaritiesPerItemKeyComparator() {
      super(CapSimilaritiesPerItemKeyWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      CapSimilaritiesPerItemKeyWritable capKey1 = (CapSimilaritiesPerItemKeyWritable) a;
      CapSimilaritiesPerItemKeyWritable capKey2 = (CapSimilaritiesPerItemKeyWritable) b;

      int result = compare(capKey1.getItemID(), capKey2.getItemID());
      if (result == 0) {
        result = -1 * compare(capKey1.getAssociatedSimilarity(), capKey2.getAssociatedSimilarity());
      }
      return result;
    }

    protected static int compare(long a, long b) {
      return (a == b) ? 0 : (a < b) ? -1 : 1;
    }

    protected static int compare(double a, double b) {
      return (a == b) ? 0 : (a < b) ? -1 : 1;
    }
  }

  public static class CapSimilaritiesPerItemKeyPartitioner
      extends Partitioner<CapSimilaritiesPerItemKeyWritable,SimilarItemWritable> {

    @Override
    public int getPartition(CapSimilaritiesPerItemKeyWritable key, SimilarItemWritable value, int numPartitions) {
      return (key.hashCode() * 127) % numPartitions;
    }
  }


  public static class CapSimilaritiesPerItemKeyGroupingComparator extends WritableComparator {

    public CapSimilaritiesPerItemKeyGroupingComparator() {
      super(CapSimilaritiesPerItemKeyWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      CapSimilaritiesPerItemKeyWritable capKey1 = (CapSimilaritiesPerItemKeyWritable) a;
      CapSimilaritiesPerItemKeyWritable capKey2 = (CapSimilaritiesPerItemKeyWritable) b;
      return a.compareTo(b);
    }
  }
}
