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

package org.apache.mahout.clustering.meanshift;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;

import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.list.IntArrayList;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according
 * to the application of some distance metric, and a point total which is the sum of all the points and is
 * used to compute the centroid when needed.
 */
public class MeanShiftCanopy extends Cluster {

  private static final Type VECTOR_TYPE = new TypeToken<Vector>() {
  }.getType();

  // TODO: this is still problematic from a scalability perspective, but how else to encode membership?
  private IntArrayList boundPoints = new IntArrayList();

  /**
   * Used for Writable
   */
  public MeanShiftCanopy() {
  }

  /**
   * Create a new Canopy containing the given point
   * 
   * @param point
   *          a Vector
   */
  public MeanShiftCanopy(Vector point, int id) {
    super(point, id);
    boundPoints.add(id);
  }

  /**
   * Create a new Canopy containing the given point, id and bound points
   * 
   * @param point
   *          a Vector
   * @param id
   *          an int identifying the canopy local to this process only
   * @param boundPoints
   *          a IntArrayList containing points ids bound to the canopy
   * @param converged
   *          true if the canopy has converged
   */
  MeanShiftCanopy(Vector point, int id, IntArrayList boundPoints, boolean converged) {
    this.setId(id);
    this.setCenter(point);
    this.setRadius(point.like());
    this.setNumPoints(1);
    this.boundPoints = boundPoints;
    setConverged(converged);
  }

  public IntArrayList getBoundPoints() {
    return boundPoints;
  }

  /**
   * The receiver overlaps the given canopy. Add my bound points to it.
   * 
   * @param canopy
   *          an existing MeanShiftCanopy
   */
  void merge(MeanShiftCanopy canopy) {
    boundPoints.addAllOf(canopy.boundPoints);
  }

  /**
   * The receiver touches the given canopy. Add respective centers.
   * 
   * @param canopy
   *          an existing MeanShiftCanopy
   */
  void touch(MeanShiftCanopy canopy) {
    canopy.observe(getCenter(), boundPoints.size());
    observe(canopy.getCenter(), canopy.boundPoints.size());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int numpoints = in.readInt();
    this.boundPoints = new IntArrayList();
    for (int i = 0; i < numpoints; i++) {
      this.boundPoints.add(in.readInt());
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(boundPoints.size());
    for (int v : boundPoints.elements()) {
      out.writeInt(v);
    }
  }

  public MeanShiftCanopy shallowCopy() {
    MeanShiftCanopy result = new MeanShiftCanopy();
    result.setId(this.getId());
    result.setCenter(this.getCenter());
    result.setRadius(this.getRadius());
    result.setNumPoints(this.getNumPoints());
    result.boundPoints = this.boundPoints;
    return result;
  }

  @Override
  public String asFormatString() {
    return formatCanopy(this);
  }

  public void setBoundPoints(IntArrayList boundPoints) {
    this.boundPoints = boundPoints;
  }

  public String getIdentifier() {
    return (isConverged() ? "MSV-" : "MSC-") + getId();
  }

  /** Format the canopy for output */
  public static String formatCanopy(MeanShiftCanopy canopy) {
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(VECTOR_TYPE, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.toJson(canopy, MeanShiftCanopy.class);
  }

  /**
   * Decodes and returns a Canopy from the formattedString
   * 
   * @param formattedString
   *          a String produced by formatCanopy
   * @return a new Canopy
   */
  public static MeanShiftCanopy decodeCanopy(String formattedString) {
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(VECTOR_TYPE, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.fromJson(formattedString, MeanShiftCanopy.class);
  }

}
