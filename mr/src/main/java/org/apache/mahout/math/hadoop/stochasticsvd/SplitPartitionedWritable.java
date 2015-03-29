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
package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Varint;

/**
 * a key for vectors allowing to identify them by their coordinates in original
 * split of A.
 * 
 * We assume all passes over A results in the same splits, thus, we can always
 * prepare side files that come into contact with A, sp that they are sorted and
 * partitioned same way.
 * <P>
 * 
 * Hashcode is defined the way that all records of the same split go to the same
 * reducer.
 * <P>
 * 
 * In addition, we are defining a grouping comparator allowing group one split
 * into the same reducer group.
 * <P>
 * 
 */
public class SplitPartitionedWritable implements
    WritableComparable<SplitPartitionedWritable> {

  private int taskId;
  private long taskItemOrdinal;

  public SplitPartitionedWritable(Mapper<?, ?, ?, ?>.Context mapperContext) {
    // this is basically a split # if i understand it right
    taskId = mapperContext.getTaskAttemptID().getTaskID().getId();
  }

  public SplitPartitionedWritable() {
  }

  public int getTaskId() {
    return taskId;
  }

  public long getTaskItemOrdinal() {
    return taskItemOrdinal;
  }

  public void incrementItemOrdinal() {
    taskItemOrdinal++;
  }

  public void setTaskItemOrdinal(long taskItemOrdinal) {
    this.taskItemOrdinal = taskItemOrdinal;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    taskId = Varint.readUnsignedVarInt(in);
    taskItemOrdinal = Varint.readUnsignedVarLong(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(taskId, out);
    Varint.writeUnsignedVarLong(taskItemOrdinal, out);
  }

  @Override
  public int hashCode() {
    int prime = 31;
    int result = 1;
    result = prime * result + taskId;
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    SplitPartitionedWritable other = (SplitPartitionedWritable) obj;
    return taskId == other.taskId;
  }

  @Override
  public int compareTo(SplitPartitionedWritable o) {
    if (taskId < o.taskId) {
      return -1;
    }
    if (taskId > o.taskId) {
      return 1;
    }
    if (taskItemOrdinal < o.taskItemOrdinal) {
      return -1;
    }
    if (taskItemOrdinal > o.taskItemOrdinal) {
      return 1;
    }
    return 0;
  }

  public static final class SplitGroupingComparator extends WritableComparator implements Serializable {

    public SplitGroupingComparator() {
      super(SplitPartitionedWritable.class, true);
    }

    @Override
    public int compare(Object a, Object b) {
      SplitPartitionedWritable o1 = (SplitPartitionedWritable) a;
      SplitPartitionedWritable o2 = (SplitPartitionedWritable) b;

      if (o1.taskId < o2.taskId) {
        return -1;
      }
      if (o1.taskId > o2.taskId) {
        return 1;
      }
      return 0;
    }

  }

}
