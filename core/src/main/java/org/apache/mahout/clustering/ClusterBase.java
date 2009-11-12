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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public abstract class ClusterBase implements Writable {

  // this cluster's clusterId
  private int id;

  // the current cluster center
  private Vector center = new SparseVector(0);

  // the number of points in the cluster
  private int numPoints = 0;

  // the Vector total of all points added to the cluster
  private Vector pointTotal = null;

  public int getId() {
    return id;
  }

  protected void setId(int id) {
    this.id = id;
  }

  public Vector getCenter() {
    return center;
  }

  protected void setCenter(Vector center) {
    this.center = center;
  }

  public int getNumPoints() {
    return numPoints;
  }

  protected void setNumPoints(int numPoints) {
    this.numPoints = numPoints;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  protected void setPointTotal(Vector pointTotal) {
    this.pointTotal = pointTotal;
  }

  public abstract String asFormatString();

  /**
   * Simply writes out the id, and that's it!
   *
   * @param out The {@link java.io.DataOutput}
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
  }

  /** Reads in the id, nothing else */
  @Override
  public void readFields(DataInput in) throws IOException {
    id = in.readInt();
  }
}
