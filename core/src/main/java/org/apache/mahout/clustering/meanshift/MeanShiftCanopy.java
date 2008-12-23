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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.matrix.CardinalityException;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.PlusFunction;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.EuclideanDistanceMeasure;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This class models a canopy as a center point, the number of points that are
 * contained within it according to the application of some distance metric, and
 * a point total which is the sum of all the points and is used to compute the
 * centroid when needed.
 */
public class MeanShiftCanopy {

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";

  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";

  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";

  public static final String CANOPY_PATH_KEY = "org.apache.mahout.clustering.canopy.path";

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

  // this canopy's canopyId
  private int canopyId;

  // the current center
  private Vector center = null;

  // the number of points in the canopy
  private int numPoints = 0;

  // the total of all points added to the canopy
  private Vector pointTotal = null;

  private List<Vector> boundPoints = new ArrayList<Vector>();

  private boolean converged = false;

  /**
   * Configure the Canopy and its distance measure
   * 
   * @param job the JobConf for this job
   */
  public static void configure(JobConf job) {
    try {
      measure = Class.forName(job.get(DISTANCE_MEASURE_KEY)).asSubclass(DistanceMeasure.class).newInstance();
      measure.configure(job);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
    nextCanopyId = 0;
    t1 = Double.parseDouble(job.get(T1_KEY));
    t2 = Double.parseDouble(job.get(T2_KEY));
    convergenceDelta = Double.parseDouble(job.get(CLUSTER_CONVERGENCE_KEY));
  }

  /**
   * Configure the Canopy for unit tests
   * 
   * @param aMeasure
   * @param aT1
   * @param aT2
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
   * Merge the given canopy into the canopies list. If it touches any existing
   * canopy (norm<T1) then add the center of each to the other. If it covers
   * any other canopies (norm<T2), then merge the given canopy with the closest
   * covering canopy. If the given canopy does not cover any other canopies, add
   * it to the canopies list.
   * 
   * @param aCanopy a MeanShiftCanopy to be merged
   * @param canopies the List<Canopy> to be appended
   */
  public static void mergeCanopy(MeanShiftCanopy aCanopy, List<MeanShiftCanopy> canopies) {
    MeanShiftCanopy closestCoveringCanopy = null;
    double closestNorm = Double.MAX_VALUE;
    for (MeanShiftCanopy canopy : canopies) {
      double norm = measure.distance(canopy.getCenter(), aCanopy.getCenter());
      if (norm < t1)
        aCanopy.touch(canopy);
      if (norm < t2)
        if (closestCoveringCanopy == null || norm < closestNorm) {
          closestNorm = norm;
          closestCoveringCanopy = canopy;
        }
    }
    if (closestCoveringCanopy == null)
      canopies.add(aCanopy);
    else
      closestCoveringCanopy.merge(aCanopy);
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests
   * and to emit the point and its covering canopies to the output. The
   * CanopyCombiner will then sum the canopy points and produce the centroids.
   * 
   * @param aCanopy a MeanShiftCanopy to be merged
   * @param canopies the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public static void mergeCanopy(MeanShiftCanopy aCanopy,
      List<MeanShiftCanopy> canopies,
      OutputCollector<Text, WritableComparable<?>> collector) throws IOException {
    MeanShiftCanopy closestCoveringCanopy = null;
    double closestNorm = 0;
    for (MeanShiftCanopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter(), aCanopy.getCenter());
      if (dist < t1)
        aCanopy.touch(collector, canopy);
      if (dist < t2)
        if (closestCoveringCanopy == null || dist < closestNorm) {
          closestCoveringCanopy = canopy;
          closestNorm = dist;
        }
    }
    if (closestCoveringCanopy == null) {
      canopies.add(aCanopy);
      aCanopy.emitCanopy(aCanopy, collector);
    } else
      closestCoveringCanopy.merge(aCanopy, collector);
  }

  /**
   * Format the canopy for output
   * 
   * @param canopy
   * @return
   */
  public static String formatCanopy(MeanShiftCanopy canopy) {
    StringBuilder builder = new StringBuilder();
    builder.append(canopy.getIdentifier()).append(" - ").append(
        canopy.getCenter().asWritableComparable().toString()).append(": ");
    for (Vector bound : canopy.boundPoints)
      builder.append(bound.asWritableComparable().toString());
    return builder.toString();
  }

  /**
   * Decodes and returns a Canopy from the formattedString
   * 
   * @param formattedString a String produced by formatCanopy
   * @return a new Canopy
   */
  public static MeanShiftCanopy decodeCanopy(String formattedString) {
    int beginIndex = formattedString.indexOf('[');
    int endIndex = formattedString.indexOf(':', beginIndex);
    String id = formattedString.substring(0, beginIndex);
    String centroid = formattedString.substring(beginIndex, endIndex);
    String boundPoints = formattedString.substring(endIndex + 1).trim();
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    if (firstChar == 'C' || startsWithV) {
      int canopyId = Integer.parseInt(formattedString.substring(1, beginIndex - 3));
      Vector canopyCentroid = DenseVector.decodeFormat(new Text(centroid));
      List<Vector> canopyBoundPoints = new ArrayList<Vector>();
      while (boundPoints.length() > 0) {
        int ix = boundPoints.indexOf(']');
        Vector v = DenseVector.decodeFormat(new Text(boundPoints.substring(0,
            ix + 1)));
        canopyBoundPoints.add(v);
        boundPoints = boundPoints.substring(ix + 1);
      }
      return new MeanShiftCanopy(canopyCentroid, canopyId, canopyBoundPoints,
          startsWithV);
    }
    return null;
  }

  /**
   * Create a new Canopy with the given canopyId
   * 
   * @param id
   */
  public MeanShiftCanopy(String id) {
    this.canopyId = Integer.parseInt(id.substring(1));
    this.center = null;
    this.pointTotal = null;
    this.numPoints = 0;
  }

  /**
   * Create a new Canopy containing the given point
   * 
   * @param point a Vector
   */
  public MeanShiftCanopy(Vector point) {
    this.canopyId = nextCanopyId++;
    this.center = point;
    this.pointTotal = point.copy();
    this.numPoints = 1;
    this.boundPoints.add(point);
  }

  /**
   * Create a new Canopy containing the given point, canopyId and bound points
   * 
   * @param point a Vector
   * @param canopyId an int identifying the canopy local to this process only
   * @param boundPoints a List<Vector> containing points bound to the canopy
   * @param converged true if the canopy has converged
   */
  MeanShiftCanopy(Vector point, int canopyId, List<Vector> boundPoints,
      boolean converged) {
    this.canopyId = canopyId;
    this.center = point;
    this.pointTotal = point.copy();
    this.numPoints = 1;
    this.boundPoints = boundPoints;
    this.converged = converged;
  }

  /**
   * Add a point to the canopy some number of times
   * 
   * @param point a Vector to add
   * @param nPoints the number of times to add the point
   * @throws CardinalityException if the cardinalities disagree
   */
  void addPoints(Vector point, int nPoints) {
    numPoints += nPoints;
    Vector subTotal = (nPoints == 1) ? point.copy() : point.times(nPoints);
    pointTotal = (pointTotal == null) ? subTotal : pointTotal.plus(subTotal);
  }

  /**
   * Return if the point is closely covered by this canopy
   * 
   * @param point a Vector point
   * @return if the point is covered
   */
  public boolean closelyBound(Vector point) {
    return measure.distance(center, point) < t2;
  }

  /**
   * Compute the bound centroid by averaging the bound points
   * 
   * @return a Vector which is the new bound centroid
   */
  public Vector computeBoundCentroid() {
    Vector result = new DenseVector(center.cardinality());
    for (Vector v : boundPoints)
      result.assign(v, new PlusFunction());
    return result.divide(boundPoints.size());
  }

  /**
   * Compute the centroid by normalizing the pointTotal
   * 
   * @return a Vector which is the new centroid
   */
  public Vector computeCentroid() {
    if (numPoints == 0)
      return center;
    else
      return pointTotal.divide(numPoints);
  }

  /**
   * Return if the point is covered by this canopy
   * 
   * @param point a Vector point
   * @return if the point is covered
   */
  boolean covers(Vector point) {
    return measure.distance(center, point) < t1;
  }

  /**
   * Emit the new canopy to the collector, keyed by the canopy's Id
   */
  void emitCanopy(MeanShiftCanopy canopy,
      OutputCollector<Text, WritableComparable<?>> collector) throws IOException {
    String identifier = this.getIdentifier();
    collector.collect(new Text(identifier),
        new Text("new " + canopy.toString()));
  }

  /**
   * Emit the canopy centroid to the collector, keyed by the canopy's Id, once
   * per bound point.
   * 
   * @param canopy a MeanShiftCanopy
   * @param collector the OutputCollector
   * @throws IOException if there is an IO problem with the collector
   */
  void emitCanopyCentroid(MeanShiftCanopy canopy,
      OutputCollector<Text, WritableComparable<?>> collector) throws IOException {
    collector.collect(new Text(this.getIdentifier()), new Text(canopy
        .computeCentroid().asWritableComparable().toString()
        + boundPoints.size()));
  }

  public List<Vector> getBoundPoints() {
    return boundPoints;
  }

  public int getCanopyId() {
    return canopyId;
  }

  /**
   * Return the center point
   * 
   * @return a Vector
   */
  public Vector getCenter() {
    return center;
  }

  public String getIdentifier() {
    return converged ? "V" + canopyId : "C" + canopyId;
  }

  /**
   * Return the number of points under the Canopy
   * 
   * @return
   */
  public int getNumPoints() {
    return numPoints;
  }

  void init(MeanShiftCanopy canopy) {
    canopyId = canopy.canopyId;
    center = canopy.center;
    addPoints(center, 1);
    boundPoints.addAll(canopy.getBoundPoints());
  }

  public boolean isConverged() {
    return converged;
  }

  /**
   * The receiver overlaps the given canopy. Touch it and add my bound points to
   * it.
   * 
   * @param canopy an existing MeanShiftCanopy
   */
  void merge(MeanShiftCanopy canopy) {
    boundPoints.addAll(canopy.boundPoints);
  }

  /**
   * The receiver overlaps the given canopy. Touch it and add my bound points to
   * it.
   * 
   * @param canopy an existing MeanShiftCanopy
   */
  void merge(MeanShiftCanopy canopy,
      OutputCollector<Text, WritableComparable<?>> collector) throws IOException {
    collector.collect(new Text(getIdentifier()), new Text("merge "
        + canopy.toString()));
  }

  public boolean shiftToMean() {
    Vector centroid = computeCentroid();
    converged = new EuclideanDistanceMeasure().distance(centroid, center) < convergenceDelta;
    center = centroid;
    numPoints = 1;
    pointTotal = centroid.copy();
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
    addPoints(canopy.center, canopy.boundPoints.size());
  }

  /**
   * The receiver touches the given canopy. Emit the respective centers.
   * 
   * @param collector
   * @param canopy
   * @throws IOException
   */
  void touch(OutputCollector<Text, WritableComparable<?>> collector,
      MeanShiftCanopy canopy) throws IOException {
    canopy.emitCanopyCentroid(this, collector);
    emitCanopyCentroid(canopy, collector);
  }
}
