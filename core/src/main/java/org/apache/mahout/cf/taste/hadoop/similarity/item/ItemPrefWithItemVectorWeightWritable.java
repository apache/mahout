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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.common.RandomUtils;

/**
 * A {@link Writable} encapsulating the preference for an item
 * stored along with the length of the item-vector
 *
 */
public final class ItemPrefWithItemVectorWeightWritable implements Writable, Cloneable {

  private EntityPrefWritable itemPref;
  private double weight;

  public ItemPrefWithItemVectorWeightWritable() {
  // do nothing
  }

  public ItemPrefWithItemVectorWeightWritable(long itemID, double weight, float prefValue) {
    this.itemPref = new EntityPrefWritable(itemID, prefValue);
    this.weight = weight;
  }

  public long getItemID() {
    return itemPref.getID();
  }

  public double getWeight() {
    return weight;
  }

  public float getPrefValue() {
    return itemPref.getPrefValue();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    itemPref.write(out);
    out.writeDouble(weight);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemPref = new EntityPrefWritable();
    itemPref.readFields(in);
    weight = in.readDouble();
  }

  @Override
  public int hashCode() {
    return itemPref.hashCode() + 31 * RandomUtils.hashDouble(weight);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof ItemPrefWithItemVectorWeightWritable) {
      ItemPrefWithItemVectorWeightWritable other = (ItemPrefWithItemVectorWeightWritable) o;
      return itemPref.equals(other.itemPref) && weight == other.getWeight();
    }
    return false;
  }

  @Override
  public ItemPrefWithItemVectorWeightWritable clone() {
    return new ItemPrefWithItemVectorWeightWritable(itemPref.getID(), weight, itemPref.getPrefValue());
  }

}
