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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/** A {@link WritableComparable} encapsulating two items. */
public final class ItemItemWritable implements WritableComparable<ItemItemWritable> {

  private long itemAID;
  private long itemBID;

  public ItemItemWritable() {
    // do nothing
  }

  public ItemItemWritable(long itemAID, long itemBID) {
    this.itemAID = itemAID;
    this.itemBID = itemBID;
  }

  public long getItemAID() {
    return itemAID;
  }

  public long getItemBID() {
    return itemBID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(itemAID);
    out.writeLong(itemBID);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemAID = in.readLong();
    itemBID = in.readLong();
  }

  public static ItemItemWritable read(DataInput in) throws IOException {
    ItemItemWritable writable = new ItemItemWritable();
    writable.readFields(in);
    return writable;
  }

  @Override
  public int compareTo(ItemItemWritable that) {
    if (this == that) {
      return 0;
    }
    if (itemAID < that.itemAID) {
      return -1;
    } else if (itemAID > that.itemAID) {
      return 1;
    } else {
      return itemBID < that.itemBID ? -1 : itemBID > that.itemBID ? 1 : 0;
    }
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(itemAID) + 31 * RandomUtils.hashLong(itemBID);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof ItemItemWritable) {
      ItemItemWritable that = (ItemItemWritable) o;
      return itemAID == that.itemAID && itemBID == that.itemBID;
    }
    return false;
  }

  @Override
  public String toString() {
    return itemAID + "\t" + itemBID;
  }

}