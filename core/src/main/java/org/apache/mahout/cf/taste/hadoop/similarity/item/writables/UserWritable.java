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
import org.apache.mahout.common.RandomUtils;

/** A {@link WritableComparable} encapsulating a user ID. */
public final class UserWritable implements WritableComparable<UserWritable> {

  private long userID;

  public UserWritable() {
    // do nothing
  }

  public UserWritable(long userID) {
    this.userID = userID;
  }

  public long getUserID() {
    return userID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeLong(userID);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    userID = in.readLong();
  }

  @Override
  public int compareTo(UserWritable other) {
    return compare(userID, other.getUserID());
  }

  private static int compare(long a, long b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(userID);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof UserWritable) {
      return (userID == ((UserWritable) o).getUserID());
    }
    return false;
  }

  public static UserWritable read(DataInput in) throws IOException {
    UserWritable writable = new UserWritable();
    writable.readFields(in);
    return writable;
  }

}
