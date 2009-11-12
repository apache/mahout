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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.CardinalityException;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.JsonVectorAdapter;
import org.apache.mahout.matrix.PlusFunction;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;

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

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";

  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";

  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";

  public static final String CONTROL_PATH_KEY = "org.apache.mahout.clustering.control.path";

  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.canopy.convergence";

  private static double convergenceDelta = 0;

  // the next canopyId to be allocated
  private static int nextCanopyId = 0;

  // the T1 distance threshold
  private static double t1;

  // the T2 distance threshold
  private static double t2;

  // the distance measure
  private static DistanceMeasure measure;

  // TODO: this is problematic, but how else to encode membership?
  private List<Vector> boundPoints = new ArrayList<Vector>();

  private boolean converged = false;

  static double getT1() {
    return t1;
  }

  static double getT2() {
    return t2;
  }

  /**
   * Configure the Canopy and its distance measure
   *
   * @param job the JobConf for this job
   */
  public static void configure(JobConf job) {
    try {
      measure = Class.forName(job.get(DISTANCE_MEASURE_KEY)).asSubclass(
          DistanceMeasure.class).newInstance();
      measure.configure(job);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    nextCanopyId = 0;
    t1 = Double.parseDouble(job.get(T1_KEY));
    t2 = Double.parseDouble(job.get(T2_KEY));
    convergenceDelta = Double.parseDouble(job.get(CLUSTER_CONVERGENCE_KEY));
  }

  /**
   * Configure the Canopy for unit tests
   *
   * @param aDelta the convergence criteria
   */
  public static void config(DistanceMeasure aMeasure, double aT1, double aT2,
                            double aDelta) {
    nextCanopyId = 100; // so canopyIds will sort properly
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
    convergenceDelta = aDelta;
  }

  /**
   * Merge the given canopy into the canopies list. If it touches any existing canopy (norm<T1) then add the center of
   * each to the other. If it covers any other canopies (norm<T2), then merge the given canopy with the closest covering
   * canopy. If the given canopy does not cover any other canopies, add it to the canopies list.
   *
   * @param aCanopy  a MeanShiftCanopy to be merged
   * @param canopies the List<Canopy> to be appended
   */
  public static void mergeCanopy(MeanShiftCanopy aCanopy,
                                 List<MeanShiftCanopy> canopies) {
    MeanShiftCanopy closestCoveringCanopy = null;
    double closestNorm = Double.MAX_VALUE;
    for (MeanShiftCanopy canopy : canopies) {
      double norm = measure.distance(canopy.getCenter(), aCanopy.getCenter());
      if (norm < t1) {
        aCanopy.touch(canopy);
      }
      if (norm < t2) {
        if (closestCoveringCanopy == null || norm < closestNorm) {
          closestNorm = norm;
          closestCoveringCanopy = canopy;
        }
      }
    }
    if (closestCoveringCanopy == null) {
      canopies.add(aCanopy);
    } else {
      closestCoveringCanopy.merge(aCanopy);
    }
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

  public MeanShiftCanopy() {
    super();
  }

  /** Create a new Canopy with the given canopyId */
  public MeanShiftCanopy(String id) {
    this.setId(Integer.parseInt(id.substring(1)));
    this.setCenter(null);
    this.setPointTotal(null);
    this.setNumPoints(0);
  }

  /**
   * Create a new Canopy containing the given point
   *
   * @param point a Vector
   */
  public MeanShiftCanopy(Vector point) {
    this.setId(nextCanopyId++);
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
   * Return if the point is closely covered by this canopy
   *
   * @param point a Vector point
   * @return if the point is covered
   */
  public boolean closelyBound(Vector point) {
    return measure.distance(getCenter(), point) < t2;
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

  /**
   * Return if the point is covered by this canopy
   *
   * @param point a Vector point
   * @return if the point is covered
   */
  boolean covers(Vector point) {
    return measure.distance(getCenter(), point) < t1;
  }

  /** Emit the new canopy to the collector, keyed by the canopy's Id */
  void emitCanopy(MeanShiftCanopy canopy,
                  OutputCollector<Text, WritableComparable<?>> collector)
      throws IOException {
    String identifier = this.getIdentifier();
    collector.collect(new Text(identifier),
        new Text("new " + canopy.toString()));
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

  /**
   * Shift the center to the new centroid of the cluster
   *
   * @return if the cluster is converged
   */
  public boolean shiftToMean() {
    Vector centroid = computeCentroid();
    converged = new EuclideanDistanceMeasure().distance(centroid, getCenter()) < convergenceDelta;
    setCenter(centroid);
    setNumPoints(1);
    setPointTotal(centroid.clone());
    return converged;
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
    this.setCenter(AbstractVector.readVector(in));
    int numpoints = in.readInt();
    this.boundPoints = new ArrayList<Vector>();
    for (int i = 0; i < numpoints; i++) {
      this.boundPoints.add(AbstractVector.readVector(in));
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    AbstractVector.writeVector(out, computeCentroid());
    out.writeInt(boundPoints.size());
    for (Vector v : boundPoints) {
      AbstractVector.writeVector(out, v);
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
}
