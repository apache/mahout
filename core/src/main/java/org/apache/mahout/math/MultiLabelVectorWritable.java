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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Writable to handle serialization of a vector and a variable list of
 * associated label indexes.
 */
public class MultiLabelVectorWritable extends VectorWritable {

  private int[] labels;

  public void setLabels(int[] labels) {
    this.labels = labels;
  }

  public int[] getLabels() {
    return labels;
  }

  public MultiLabelVectorWritable() {}

  public MultiLabelVectorWritable(Vector v, int[] labels) {
    super(v);
    setLabels(labels);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int labelSize = in.readInt();
    labels = new int[labelSize];
    for (int i = 0; i < labelSize; i++) {
      labels[i] = in.readInt();
    }
    super.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(labels.length);
    for (int i = 0; i < labels.length; i++) {
      out.writeInt(labels[i]);
    }
    super.write(out);
  }

  public static MultiLabelVectorWritable read(DataInput in) throws IOException {
    int labelSize = in.readInt();
    int[] labels = new int[labelSize];
    for (int i = 0; i < labelSize; i++) {
      labels[i] = in.readInt();
    }
    Vector vector = VectorWritable.readVector(in);
    return new MultiLabelVectorWritable(vector, labels);
  }

  public static void write(DataOutput out, SequentialAccessSparseVector ssv,
      int[] labels) throws IOException {
    (new MultiLabelVectorWritable(ssv, labels)).write(out);
  }

}
