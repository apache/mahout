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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.PlusFunction;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according to the
 * application of some distance metric, and a point total which is the sum of all the points and is used to compute the
 * centroid when needed.
 */
public class MeanShiftCanopy extends ClusterBase {

   // TODO: this is problematic, but how else to encode membership?
  private List<Vector> boundPoints = new ArrayList<Vector>();

  private boolean converged = false;

  public MeanShiftCanopy() {
    super();
  }

  /** Create a new Canopy with the given canopyId */
  /*
  public MeanShiftCanopy(String id) {
    this.setId(Integer.parseInt(id.substring(1)));
    this.setCenter(null);
    this.setPointTotal(null);
    this.setNumPoints(0);
  }
  */

  /**
   * Create a new Canopy containing the given point
   *
   * @param point a Vector
   */
  /*
  public MeanShiftCanopy(Vector point) {
    this.setCenter(point);
    this.setPointTotal(point.clone());
    this.setNumPoints(1);
    this.boundPoints.add(point);
  }
  */

  /**
   * Create a new Canopy containing the given point
   *
   * @param point a Vector
   */
  public MeanShiftCanopy(Vector point, int id) {
    this.setId(id);
    this.setCenter(point);
    this.setPointTotal(point.clone());
    this.setNumPoints(1);
    this.boundPoints.add(point);
  }
  
  /**
   * Create a new Canopy containing the given point, id and bound points
   *
   * @param point       a Vector
   * @param id          an int identifying the canopy local to this process only
   * @param boundPoints a List<Vector> containing points bound to the canopy
   * @param converged   true if the canopy has converged
   */
  MeanShiftCanopy(Vector point, int id, List<Vector> boundPoints,
                  boolean converged) {
    this.setId(id);
    this.setCenter(point);
    this.setPointTotal(point.clone());
    this.setNumPoints(1);
    this.boundPoints = boundPoints;
    this.converged = converged;
  }

  /**
   * Add a point to the canopy some number of times
   *
   * @param point   a Vector to add
   * @param nPoints the number of times to add the point
   * @throws CardinalityException if the cardinalities disagree
   */
  void addPoints(Vector point, int nPoints) {
    setNumPoints(getNumPoints() + nPoints);
    Vector subTotal = (nPoints == 1) ? point.clone() : point.times(nPoints);
    setPointTotal((getPointTotal() == null) ? subTotal : getPointTotal().plus(subTotal));
  }

  /**
   * Compute the bound centroid by averaging the bound points
   *
   * @return a Vector which is the new bound centroid
   */
  public Vector computeBoundCentroid() {
    Vector result = new DenseVector(getCenter().size());
    for (Vector v : boundPoints) {
      result.assign(v, new PlusFunction());
    }
    return result.divide(boundPoints.size());
  }

  /**
   * Compute the centroid by normalizing the pointTotal
   *
   * @return a Vector which is the new centroid
   */
  public Vector computeCentroid() {
    if (getNumPoints() == 0) {
      return getCenter();
    } else {
      return getPointTotal().divide(getNumPoints());
    }
  }

  public List<Vector> getBoundPoints() {
    return boundPoints;
  }

  public int getCanopyId() {
    return getId();
  }

  public String getIdentifier() {
    return converged ? "V" + getId() : "C" + getId();
  }

  void init(MeanShiftCanopy canopy) {
    setId(canopy.getId());
    setCenter(canopy.getCenter());
    addPoints(getCenter(), 1);
    boundPoints.addAll(canopy.getBoundPoints());
  }

  public boolean isConverged() {
    return converged;
  }

  /**
   * The receiver overlaps the given canopy. Touch it and add my bound points to it.
   *
   * @param canopy an existing MeanShiftCanopy
   */
  void merge(MeanShiftCanopy canopy) {
    boundPoints.addAll(canopy.boundPoints);
  }

  @Override
  public String toString() {
    return formatCanopy(this);
  }

  /**
   * The receiver touches the given canopy. Add respective centers.
   *
   * @param canopy an existing MeanShiftCanopy
   */
  void touch(MeanShiftCanopy canopy) {
    canopy.addPoints(getCenter(), boundPoints.size());
    addPoints(canopy.getCenter(), canopy.boundPoints.size());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    this.setCenter(VectorWritable.readVector(in));
    int numpoints = in.readInt();
    this.boundPoints = new ArrayList<Vector>();
    for (int i = 0; i < numpoints; i++) {
      this.boundPoints.add(VectorWritable.readVector(in));
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    VectorWritable.writeVector(out, computeCentroid());
    out.writeInt(boundPoints.size());
    for (Vector v : boundPoints) {
      VectorWritable.writeVector(out, v);
    }
  }

  public MeanShiftCanopy shallowCopy() {
    MeanShiftCanopy result = new MeanShiftCanopy();
    result.setId(this.getId());
    result.setCenter(this.getCenter());
    result.setPointTotal(this.getPointTotal());
    result.setNumPoints(this.getNumPoints());
    result.boundPoints = this.boundPoints;
    return result;
  }

  @Override
  public String asFormatString() {
    return formatCanopy(this);
  }
  
  public void setBoundPoints(List<Vector> boundPoints) {
    this.boundPoints = boundPoints;
  }

  public void setConverged(boolean converged) {
    this.converged = converged;
  }

  /** Format the canopy for output */
  public static String formatCanopy(MeanShiftCanopy canopy) {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.toJson(canopy, MeanShiftCanopy.class);
  }

  /**
   * Decodes and returns a Canopy from the formattedString
   *
   * @param formattedString a String produced by formatCanopy
   * @return a new Canopy
   */
  public static MeanShiftCanopy decodeCanopy(String formattedString) {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    GsonBuilder gBuilder = new GsonBuilder();
    gBuilder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = gBuilder.create();
    return gson.fromJson(formattedString, MeanShiftCanopy.class);
  }

}
