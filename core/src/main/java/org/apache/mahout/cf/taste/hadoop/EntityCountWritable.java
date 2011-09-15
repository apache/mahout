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

package org.apache.mahout.cf.taste.hadoop;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Varint;

/** A {@link org.apache.hadoop.io.Writable} encapsulating an item ID and a count . */
public final class EntityCountWritable extends VarLongWritable {

  private int count;

  public EntityCountWritable() {
    // do nothing
  }

  public EntityCountWritable(long itemID, int count) {
    super(itemID);
    this.count = count;
  }

  public EntityCountWritable(EntityCountWritable other) {
    this(other.get(), other.getCount());
  }

  public long getID() {
    return get();
  }

  public int getCount() {
    return count;
  }

  public void set(long id, int count) {
    set(id);
    this.count = count;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    Varint.writeUnsignedVarInt(count, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    count = Varint.readUnsignedVarInt(in);
  }

  @Override
  public int hashCode() {
    return super.hashCode() ^ count;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof EntityCountWritable)) {
      return false;
    }
    EntityCountWritable other = (EntityCountWritable) o;
    return get() == other.get() && count == other.getCount();
  }

  @Override
  public String toString() {
    return get() + "\t" + count;
  }

  @Override
  public EntityCountWritable clone() {
    return new EntityCountWritable(get(), count);
  }

}