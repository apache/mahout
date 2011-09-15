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

package org.apache.mahout.clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class ClusterObservations implements Writable {

  private int combinerState;

  private double s0;

  private Vector s1;

  private Vector s2;

  public ClusterObservations(double s0, Vector s1, Vector s2) {
    this.s0 = s0;
    this.s1 = s1;
    this.s2 = s2;
  }

  public ClusterObservations(int combinerState, double s0, Vector s1, Vector s2) {
    this.combinerState = combinerState;
    this.s0 = s0;
    this.s1 = s1;
    this.s2 = s2;
  }

  public ClusterObservations() {
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.combinerState = in.readInt();
    this.s0 = in.readDouble();
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.s1 = temp.get();
    temp.readFields(in);
    this.s2 = temp.get();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(combinerState);
    out.writeDouble(s0);
    VectorWritable.writeVector(out, s1);
    VectorWritable.writeVector(out, s2);
  }

  /**
   * @return the combinerState
   */
  public int getCombinerState() {
    return combinerState;
  }

  /**
   * @return the s0
   */
  public double getS0() {
    return s0;
  }

  /**
   * @return the s1
   */
  public Vector getS1() {
    return s1;
  }

  /**
   * @return the s2
   */
  public Vector getS2() {
    return s2;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder(50);
    buf.append("co{s0=").append(s0).append(" s1=");
    if (s1 != null) {
      buf.append(AbstractCluster.formatVector(s1, null));
    }
    buf.append(" s2=");
    if (s2 != null) {
      buf.append(AbstractCluster.formatVector(s2, null));
    }
    buf.append('}');
    return buf.toString();
  }

  public ClusterObservations incrementCombinerState() {
    combinerState++;
    return this;
  }

}
