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
import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.Vector;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * ClusterBase is an abstract base class class for several clustering implementations
 * that share common implementations of various atttributes
 *
 */
public abstract class ClusterBase implements Writable, Cluster {

  private static final Type VECTOR_TYPE = new TypeToken<Vector>() {}.getType();

  // this cluster's clusterId
  private int id;

  // the current cluster center
  private Vector center;

  // the number of points in the cluster
  private int numPoints;

  // the Vector total of all points added to the cluster
  private Vector pointTotal;

  @Override
  public int getId() {
    return id;
  }

  public void setId(int id) {
    this.id = id;
  }

  @Override
  public Vector getCenter() {
    return center;
  }

  public void setCenter(Vector center) {
    this.center = center;
  }

  @Override
  public int getNumPoints() {
    return numPoints;
  }

  public void setNumPoints(int numPoints) {
    this.numPoints = numPoints;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  public void setPointTotal(Vector pointTotal) {
    this.pointTotal = pointTotal;
  }

  /**
   * @deprecated
   * @return
   */
  @Deprecated
  public abstract String asFormatString();

  @Override
  public String asFormatString(String[] bindings) {
    StringBuilder buf = new StringBuilder();
    buf.append(getIdentifier()).append(": ").append(AbstractCluster.formatVector(getCenter(), bindings));
    return buf.toString();
  }

  public abstract Vector computeCentroid();

  public abstract Object getIdentifier();

  @Override
  public String asJsonString() {
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(VECTOR_TYPE, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.toJson(this, this.getClass());
  }

  /**
   * Simply writes out the id, and that's it!
   * 
   * @param out
   *          The {@link java.io.DataOutput}
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeInt(numPoints);
  }

  /** Reads in the id, nothing else */
  @Override
  public void readFields(DataInput in) throws IOException {
    id = in.readInt();
    numPoints = in.readInt();
  }
}
