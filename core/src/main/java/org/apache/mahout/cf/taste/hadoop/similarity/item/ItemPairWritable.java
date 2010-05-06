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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;

/**
 * A {@link WritableComparable} encapsulating two items together with their
 * multiplied vector lengths
 */
public final class ItemPairWritable implements WritableComparable<ItemPairWritable> {

  private EntityEntityWritable itemItemWritable;
  private double itemAWeight;
  private double itemBWeight;

  public ItemPairWritable() {
  }

  public ItemPairWritable(long itemAID, long itemBID, double itemAWeight, double itemBWeight) {
    this.itemItemWritable = new EntityEntityWritable(itemAID, itemBID);
    this.itemAWeight = itemAWeight;
    this.itemBWeight = itemBWeight;
  }

  public long getItemAID() {
    return itemItemWritable.getAID();
  }

  public long getItemBID() {
    return itemItemWritable.getBID();
  }

  public EntityEntityWritable getItemItemWritable() {
    return itemItemWritable;
  }

  public double getItemAWeight() {
    return itemAWeight;
  }

  public double getItemBWeight() {
    return itemBWeight;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemItemWritable = new EntityEntityWritable();
    itemItemWritable.readFields(in);
    itemAWeight = in.readDouble();
    itemBWeight = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    itemItemWritable.write(out);
    out.writeDouble(itemAWeight);
    out.writeDouble(itemBWeight);
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
