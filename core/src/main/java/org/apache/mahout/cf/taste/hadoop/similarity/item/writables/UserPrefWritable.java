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

import org.apache.hadoop.io.Writable;

/**
 * A {@link Writable} encapsulating an item ID together with a preference value.
 *
 * Used as entry in an item-vector
 */
public final class UserPrefWritable extends UserWritable {

  private float prefValue;

  public UserPrefWritable() {
  }

  public UserPrefWritable(long userID, float prefValue) {
    super(userID);
    this.prefValue = prefValue;
  }

  public float getPrefValue() {
    return prefValue;
  }

  public UserPrefWritable deepCopy() {
    return new UserPrefWritable(getUserID(), prefValue);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    prefValue =  in.readFloat();
  }

  @Override
  public void write(DataOutput out) throws IOException {
   super.write(out);
   out.writeFloat(prefValue);
  }

  @Override
  public int hashCode() {
    return super.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof UserPrefWritable) {
      UserWritable other = (UserWritable) o;
      return super.equals(other);
    }
    return false;
  }
}
