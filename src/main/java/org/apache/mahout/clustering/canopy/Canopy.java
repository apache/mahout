/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.canopy;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

/**
 * This class models a canopy as a center point, the number of points that are
 * contained within it according to the application of some distance metric, and
 * a point total which is the sum of all the points and is used to compute the
 * centroid when needed.
 * 
 */
public class Canopy {

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";

  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";

  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";

  public static final String CANOPY_PATH_KEY = "org.apache.mahout.clustering.canopy.path";

  // the next canopyId to be allocated
  private static int nextCanopyId = 0;

  // the T1 distance threshold
  private static float t1;

  // the T2 distance threshold
  private static float t2;

  // the distance measure
  private static DistanceMeasure measure;

  // this canopy's canopyId
  private int canopyId;

  // the current center
  private Float[] center = new Float[0];

  // the number of points in the canopy
  private int numPoints = 0;

  // the total of all points added to the canopy
  private Float[] pointTotal = null;

  /**
   * Create a new Canopy containing the given point
   * 
   * @param point a Float[]
   */
  public Canopy(Float[] point) {
    super();
    this.canopyId = nextCanopyId++;
    this.center = point;
    this.pointTotal = point.clone();
    this.numPoints = 1;
  }

  /**
   * Create a new Canopy containing the given point and canopyId
   * 
   * @param point a Float[]
   * @param canopyId an int identifying the canopy local to this process only
   */
  public Canopy(Float[] point, int canopyId) {
    super();
    this.canopyId = canopyId;
    this.center = point;
    this.pointTotal = point.clone();
    this.numPoints = 1;
  }

  /**
   * Configure the Canopy and its distance measure
   * 
   * @param job the JobConf for this job
   */
  public static void configure(JobConf job) {
    try {
      Class cl = Class.forName(job.get(DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);
    } catch (Exception e) {
      e.printStackTrace();
    }
    nextCanopyId = 0;
    t1 = new Float(job.get(T1_KEY));
    t2 = new Float(job.get(T2_KEY));
  }

  /**
   * Configure the Canopy for unit tests
   * @param aMeasure
   * @param aT1
   * @param aT2
   */
  public static void config(DistanceMeasure aMeasure, float aT1, float aT2) {
    nextCanopyId = 0;
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
  }

  /**
   * This is the same algorithm as the reference but inverted to iterate over
   * existing canopies instead of the points. Because of this it does not need
   * to actually store the points, instead storing a total points vector and the
   * number of points. From this a centroid can be computed.
   * 
   * This method is used by the CanopyReducer.
   * 
   * @param point the Float[] defining the point to be added
   * @param canopies the List<Canopy> to be appended
   */
  public static void addPointToCanopies(Float[] point, List<Canopy> canopies) {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      float dist = measure.distance(canopy.getCenter(), point);
      if (dist < t1)
        canopy.addPoint(point);
      pointStronglyBound = pointStronglyBound | (dist < t2);
    }
    if (!pointStronglyBound)
      canopies.add(new Canopy(point));
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests
   * and to emit the point and its covering canopies to the output. The
   * CanopyCombiner will then sum the canopy points and produce the centroids.
   * 
   * @param point the Float[] defining the point to be added
   * @param canopies the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public static void emitPointToNewCanopies(Float[] point,
      List<Canopy> canopies, OutputCollector collector) throws IOException {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      float dist = measure.distance(canopy.getCenter(), point);
      if (dist < t1)
        canopy.emitPoint(point, collector);
      pointStronglyBound = pointStronglyBound | (dist < t2);
    }
    if (!pointStronglyBound) {
      Canopy canopy = new Canopy(point);
      canopies.add(canopy);
      canopy.emitPoint(point, collector);
    }
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests
   * and to emit the point keyed by its covering canopies to the output. if the
   * point is not covered by any canopies (due to canopy centroid clustering),
   * emit the point to the closest covering canopy.
   * 
   * @param point the Float[] defining the point to be added
   * @param canopies the List<Canopy> to be appended
   * @param writable the original Writable from the input, may include arbitrary
   *        payload information after the point [...]<payload>
   * @param collector an OutputCollector in which to emit the point
   */
  public static void emitPointToExistingCanopies(Float[] point,
      List<Canopy> canopies, Writable writable, OutputCollector collector)
      throws IOException {
    float minDist = Float.MAX_VALUE;
    Canopy closest = null;
    boolean isCovered = false;
    for (Canopy canopy : canopies) {
      float dist = measure.distance(canopy.getCenter(), point);
      if (dist < t1) {
        isCovered = true;
        collector.collect(new Text(Canopy.formatCanopy(canopy)), writable);
      } else if (dist < minDist) {
        minDist = dist;
        closest = canopy;
      }
    }
    // if the point is not contained in any canopies (due to canopy centroid
    // clustering), emit the point to the closest covering canopy.
    if (!isCovered)
      collector.collect(new Text(Canopy.formatCanopy(closest)), writable);
  }

  /**
   * Returns a print string for the point
   * 
   * @param out a String to append to
   * @param pt the Float[] point
   * @return
   */
  public static String ptOut(String out, Float[] pt) {
    out += formatPoint(pt);
    return out;
  }

  /**
   * Format the point for input to a Mapper or Reducer
   * 
   * @param point a Float[]
   * @return a String
   */
  public static String formatPoint(Float[] point) {
    String out = "";
    out += "[";
    for (int i = 0; i < point.length; i++)
      out += point[i] + ", ";
    out += "] ";
    String ptOut = out;
    return ptOut;
  }

  /**
   * Decodes a point from its string representation.
   * 
   * @param formattedString a comma-terminated String of the form
   *        "[v1,v2,...,vn,]"
   * @return the Float[] defining an n-dimensional point
   */
  public static Float[] decodePoint(String formattedString) {
    String[] pts = formattedString.split(",");
    Float[] point = new Float[pts.length - 1];
    for (int i = 0; i < point.length; i++)
      if (pts[i].startsWith("["))
        point[i] = new Float(pts[i].substring(1));
      else if (!pts[i].startsWith("]"))
        point[i] = new Float(pts[i]);
    return point;
  }

  /**
   * Format the canopy for output
   * 
   * @param canopy
   * @return
   */
  public static String formatCanopy(Canopy canopy) {
    return "C" + canopy.canopyId + ": " + formatPoint(canopy.computeCentroid());
  }

  /**
   * Decodes and returns a Canopy from the formattedString
   * 
   * @param formattedString a String prouced by formatCanopy
   * @return a new Canopy
   */
  public static Canopy decodeCanopy(String formattedString) {
    int beginIndex = formattedString.indexOf('[');
    String id = formattedString.substring(0, beginIndex);
    String centroid = formattedString.substring(beginIndex);
    if (id.startsWith("C")) {
      int canopyId = new Integer(formattedString.substring(1, beginIndex - 2));
      Float[] canopyCentroid = decodePoint(centroid);
      return new Canopy(canopyCentroid, canopyId);
    }
    return null;
  }

  /**
   * Add a point to the canopy
   * 
   * @param point a Float[]
   */
  public void addPoint(Float[] point) {
    numPoints++;
    for (int i = 0; i < point.length; i++)
      pointTotal[i] = new Float(point[i] + pointTotal[i]);
  }

  /**
   * Emit the point to the collector, keyed by the canopy's formatted
   * representation
   * 
   * @param point a Float[]
   */
  public void emitPoint(Float[] point, OutputCollector collector)
      throws IOException {
    collector.collect(new Text(formatCanopy(this)), new Text(ptOut("", point)));
  }

  /**
   * Return a printable representation of this object, using the user supplied
   * identifier
   * 
   * @return
   */
  public String toString() {
    return "C" + canopyId + " - " + ptOut("", getCenter());
  }

  public int getCanopyId() {
    return canopyId;
  }

  /**
   * Return the center point
   * 
   * @return a Float[]
   */
  public Float[] getCenter() {
    return center;
  }

  /**
   * Return the number of points in the Canopy
   * 
   * @return
   */
  public int getNumPoints() {
    return numPoints;
  }

  /**
   * Compute the centroid by averaging the pointTotals
   * 
   * @return a Float[] which is the new centroid
   */
  public Float[] computeCentroid() {
    Float[] result = new Float[pointTotal.length];
    for (int i = 0; i < pointTotal.length; i++)
      result[i] = new Float(pointTotal[i] / numPoints);
    return result;
  }

  /**
   * Return if the point is covered by this canopy
   * 
   * @param point a Float[] point
   * @return if the point is covered
   */
  public boolean covers(Float[] point) {
    return measure.distance(center, point) < t1;
  }
}
