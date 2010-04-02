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

/** A {@link Writable} encapsulating an item ID. */
public class ItemWritable implements Writable {

  private long itemID;

  public ItemWritable() {
    // do nothing
  }

  public ItemWritable(long itemID) {
    this.itemID = itemID;
  }

  public ItemWritable(ItemWritable other) {
    this(other.getItemID());
  }

  public long getItemID() {
    return itemID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(itemID);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemID = in.readLong();
  }

  public static ItemWritable read(DataInput in) throws IOException {
    ItemWritable writable = new ItemWritable();
    writable.readFields(in);
    return writable;
  }

}