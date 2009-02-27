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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * A {@link WritableComparable} encapsulating two {@link org.apache.mahout.cf.taste.model.Item}s.
 */
public final class ItemItemWritable implements WritableComparable<ItemItemWritable> {

  private String itemAID;
  private String itemBID;

  public ItemItemWritable() {
    // do nothing
  }

  public ItemItemWritable(String itemAID, String itemBID) {
    this.itemAID = itemAID;
    this.itemBID = itemBID;
  }

  public String getItemAID() {
    return itemAID;
  }

  public String getItemBID() {
    return itemBID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(getItemAID());
    out.writeUTF(getItemBID());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemAID = in.readUTF();
    itemBID = in.readUTF();
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
    int compare = itemAID.compareTo(that.itemAID);
    return compare == 0 ? itemBID.compareTo(that.itemBID) : compare;
  }

  @Override
  public int hashCode() {
    return itemAID.hashCode() + 31 * itemBID.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof ItemItemWritable) {
      ItemItemWritable that = (ItemItemWritable) o;
      return this == that || (itemAID.equals(that.itemAID) && itemBID.equals(that.itemBID));
    }
    return false;
  }

  @Override
  public String toString() {
    return itemAID + '\t' + itemBID;
  }

}