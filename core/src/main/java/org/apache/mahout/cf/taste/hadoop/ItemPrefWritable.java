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

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * A {@link Writable} encapsulating a {@link org.apache.mahout.cf.taste.model.Item} and a preference value.
 */
public final class ItemPrefWritable implements Writable {

  private String itemID;
  private double prefValue;

  public ItemPrefWritable() {
    // do nothing
  }

  public ItemPrefWritable(String itemID, double prefValue) {
    this.itemID = itemID;
    this.prefValue = prefValue;
  }

  public ItemPrefWritable(ItemPrefWritable other) {
    this(other.getItemID(), other.getPrefValue());
  }

  public String getItemID() {
    return itemID;
  }

  public double getPrefValue() {
    return prefValue;
  }

  public void write(DataOutput out) throws IOException {
    out.writeUTF(itemID);
    out.writeDouble(prefValue);
  }

  public void readFields(DataInput in) throws IOException {
    itemID = in.readUTF();
    prefValue = in.readDouble();
  }

  public static ItemPrefWritable read(DataInput in) throws IOException {
    ItemPrefWritable writable = new ItemPrefWritable();
    writable.readFields(in);
    return writable;
  }

}