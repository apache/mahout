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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.common.RandomUtils;

/** A {@link Writable} encapsulating an item ID. */
public class EntityWritable implements WritableComparable<EntityWritable>, Cloneable {

  private long ID;

  public EntityWritable() {
    // do nothing
  }

  public EntityWritable(long ID) {
    this.ID = ID;
  }

  public EntityWritable(EntityWritable other) {
    this(other.getID());
  }

  public long getID() {
    return ID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(ID);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    ID = in.readLong();
  }

  public static EntityWritable read(DataInput in) throws IOException {
    EntityWritable writable = new EntityWritable();
    writable.readFields(in);
    return writable;
  }

  @Override
  public int compareTo(EntityWritable other) {
    long otherItemID = other.getID();
    return ID < otherItemID ? -1 : ID > otherItemID ? 1 : 0;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(ID);
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof EntityWritable && (ID == ((EntityWritable) o).getID());
  }

  @Override
  public EntityWritable clone() {
    return new EntityWritable(ID);
  }

}