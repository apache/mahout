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

package org.apache.mahout.math.hadoop.similarity;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob.EntriesToVectorsReducer;

/**
 * used as key for the {@link EntriesToVectorsReducer} to collect all rows similar to the specified row
 *
 * ensures that the similarity matrix entries for a row are seen in descending order
 * by their similarity value via secondary sort
 */
public class SimilarityMatrixEntryKey implements WritableComparable<SimilarityMatrixEntryKey> {

  private int row;
  private double value;

  static {
    WritableComparator.define(SimilarityMatrixEntryKey.class, new SimilarityMatrixEntryKeyComparator());
  }

  public SimilarityMatrixEntryKey() {
  }

  public SimilarityMatrixEntryKey(int row, double value) {
    this.row = row;
    this.value = value;
  }

  public void set(int row, double value) {
    this.row = row;
    this.value = value;
  }

  public int getRow() {
    return row;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    row = Varint.readSignedVarInt(in);
    value = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarInt(row, out);
    out.writeDouble(value);
  }

  @Override
  public int compareTo(SimilarityMatrixEntryKey other) {
    return (row == other.row) ? 0 : (row < other.row) ? -1 : 1;
  }

  @Override
  public int hashCode() {
    return row;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof SimilarityMatrixEntryKey && row == ((SimilarityMatrixEntryKey) other).row;
  }

  public static class SimilarityMatrixEntryKeyComparator extends WritableComparator {

    protected SimilarityMatrixEntryKeyComparator() {
      super(SimilarityMatrixEntryKey.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      SimilarityMatrixEntryKey key1 = (SimilarityMatrixEntryKey) a;
      SimilarityMatrixEntryKey key2 = (SimilarityMatrixEntryKey) b;

      int result = compare(key1.row, key2.row);
      if (result == 0) {
        result = -1 * compare(key1.value, key2.value);
      }
      return result;
    }

    protected static int compare(long a, long b) {
      return (a == b) ? 0 : (a < b) ? -1 : 1;
    }

    protected static int compare(double a, double b) {
      return (a == b) ? 0 : (a < b) ? -1 : 1;
    }
  }

  public static class SimilarityMatrixEntryKeyPartitioner
      extends Partitioner<SimilarityMatrixEntryKey,MatrixEntryWritable> {
    @Override
    public int getPartition(SimilarityMatrixEntryKey key, MatrixEntryWritable value, int numPartitions) {
      return (key.hashCode() * 127) % numPartitions;
    }
  }

  public static class SimilarityMatrixEntryKeyGroupingComparator extends WritableComparator {

    protected SimilarityMatrixEntryKeyGroupingComparator() {
      super(SimilarityMatrixEntryKey.class, true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
      return a.compareTo(b);
    }
  }

}
