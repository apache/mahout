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

package org.apache.mahout.cf.taste.hadoop.similarity.item.writables;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.cf.taste.hadoop.ItemItemWritable;

/**
 * A {@link WritableComparable} encapsulating two items together with their
 * multiplied vector lengths
 */
public final class ItemPairWritable implements WritableComparable<ItemPairWritable> {

  private ItemItemWritable itemItemWritable;
  private double multipliedLength;

  public ItemPairWritable() {
  }

  public ItemPairWritable(long itemAID, long itemBID, double multipliedLength) {
    this.itemItemWritable = new ItemItemWritable(itemAID, itemBID);
    this.multipliedLength = multipliedLength;
  }

  public long getItemAID() {
    return itemItemWritable.getItemAID();
  }

  public long getItemBID() {
    return itemItemWritable.getItemBID();
  }

  public ItemItemWritable getItemItemWritable() {
    return itemItemWritable;
  }

  public double getMultipliedLength() {
    return multipliedLength;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemItemWritable = ItemItemWritable.read(in);
    multipliedLength = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    itemItemWritable.write(out);
    out.writeDouble(multipliedLength);
  }

  @Override
  public int compareTo(ItemPairWritable other) {
    return itemItemWritable.compareTo(other.getItemItemWritable());
  }

  @Override
  public int hashCode() {
    return itemItemWritable.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof ItemPairWritable) {
      return itemItemWritable.equals(((ItemPairWritable) o).getItemItemWritable());
    }
    return false;
  }

}
