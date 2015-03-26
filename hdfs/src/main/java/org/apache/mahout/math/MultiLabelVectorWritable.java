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

package org.apache.mahout.math;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Writable to handle serialization of a vector and a variable list of
 * associated label indexes.
 */
public final class MultiLabelVectorWritable implements Writable {

  private final VectorWritable vectorWritable = new VectorWritable();
  private int[] labels;

  public MultiLabelVectorWritable() {
  }

  public MultiLabelVectorWritable(Vector vector, int[] labels) {
    this.vectorWritable.set(vector);
    this.labels = labels;
  }

  public Vector getVector() {
    return vectorWritable.get();
  }

  public void setVector(Vector vector) {
    vectorWritable.set(vector);
  }

  public void setLabels(int[] labels) {
    this.labels = labels;
  }

  public int[] getLabels() {
    return labels;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    vectorWritable.readFields(in);
    int labelSize = in.readInt();
    labels = new int[labelSize];
    for (int i = 0; i < labelSize; i++) {
      labels[i] = in.readInt();
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    vectorWritable.write(out);
    out.writeInt(labels.length);
    for (int label : labels) {
      out.writeInt(label);
    }
  }

  public static MultiLabelVectorWritable read(DataInput in) throws IOException {
    MultiLabelVectorWritable writable = new MultiLabelVectorWritable();
    writable.readFields(in);
    return writable;
  }

  public static void write(DataOutput out, SequentialAccessSparseVector ssv, int[] labels) throws IOException {
    new MultiLabelVectorWritable(ssv, labels).write(out);
  }

}
