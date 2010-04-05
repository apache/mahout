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
import org.apache.mahout.common.RandomUtils;

/** A {@link Writable} encapsulating an item ID and a preference value. */
public final class EntityPrefWritable extends EntityWritable {
  
  private float prefValue;
  
  public EntityPrefWritable() {
    // do nothing
  }
  
  public EntityPrefWritable(long itemID, float prefValue) {
    super(itemID);
    this.prefValue = prefValue;
  }
  
  public EntityPrefWritable(EntityPrefWritable other) {
    this(other.getID(), other.getPrefValue());
  }

  public float getPrefValue() {
    return prefValue;
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeFloat(prefValue);
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    prefValue = in.readFloat();
  }
  
  public static EntityPrefWritable read(DataInput in) throws IOException {
    EntityPrefWritable writable = new EntityPrefWritable();
    writable.readFields(in);
    return writable;
  }

  @Override
  public int hashCode() {
    return super.hashCode() ^ RandomUtils.hashFloat(prefValue);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof EntityPrefWritable)) {
      return false;
    }
    EntityPrefWritable other = (EntityPrefWritable) o;
    return getID() == other.getID() && prefValue == other.getPrefValue();
  }

  @Override
  public EntityPrefWritable clone() {
    return new EntityPrefWritable(getID(), prefValue);
  }
  
}