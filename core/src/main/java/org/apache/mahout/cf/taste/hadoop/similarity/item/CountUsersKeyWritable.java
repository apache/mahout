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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Varint;

/**
 * a writable key that is used by {@link CountUsersMapper} and {@link CountUsersReducer} to
 * count unique users by sending all userIDs to the same reducer and have them sorted in
 * ascending order so that there's no buffering necessary when counting them
 */
public class CountUsersKeyWritable implements WritableComparable<CountUsersKeyWritable> {

  private long userID;

  public CountUsersKeyWritable() {
  }

  public CountUsersKeyWritable(long userID) {
    this.userID = userID;
  }

  public long getUserID() {
    return userID;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    userID = Varint.readSignedVarLong(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarLong(userID, out);
  }

  @Override
  public int compareTo(CountUsersKeyWritable other) {
    return userID == other.userID ? 0 : userID < other.userID ? -1 : 1;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof CountUsersKeyWritable)) {
      return false;
    }
    return userID == ((CountUsersKeyWritable) other).userID;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(userID);
  }

  /**
   * all userIDs go to the same partition
   */
  public static class CountUsersPartitioner extends Partitioner<CountUsersKeyWritable,VarLongWritable> {

    @Override
    public int getPartition(CountUsersKeyWritable key, VarLongWritable value, int numPartitions) {
      return 0;
    }

  }

  /**
   * all userIDs go to the same reducer
   */
  public static class CountUsersGroupComparator extends WritableComparator implements Serializable {

    public CountUsersGroupComparator() {
      super(CountUsersKeyWritable.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      return 0;
    }
  }
}
