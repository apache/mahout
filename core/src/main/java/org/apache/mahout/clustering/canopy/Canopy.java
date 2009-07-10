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

package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according to the
 * application of some distance metric, and a point total which is the sum of all the points and is used to compute the
 * centroid when needed.
 */
public class Canopy extends ClusterBase implements Writable {

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";

  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";

  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";

  public static final String CANOPY_PATH_KEY = "org.apache.mahout.clustering.canopy.path";

  // the next canopyId to be allocated
  private static int nextCanopyId = 0;

  // the T1 distance threshold
  private static double t1;

  // the T2 distance threshold
  private static double t2;

  // the distance measure
  private static DistanceMeasure measure;


  /** Used w */
  public Canopy() {
  }

  /**
   * Create a new Canopy containing the given point
   *
   * @param point a point in vector space
   */
  public Canopy(Vector point) {
    this.id = nextCanopyId++;
    this.center = point.clone();
    this.pointTotal = point.clone();
    this.numPoints = 1;
  }

  /**
   * Create a new Canopy containing the given point and canopyId
   *
   * @param point    a point in vector space
   * @param canopyId an int identifying the canopy local to this process only
   */
  public Canopy(Vector point, int canopyId) {
    this.id = canopyId;
    this.center = point.clone();
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
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job.get(DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
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
  }

  /** Configure the Canopy for unit tests */
  public static void config(DistanceMeasure aMeasure, double aT1, double aT2) {
    nextCanopyId = 0;
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
  }

  /**
   * This is the same algorithm as the reference but inverted to iterate over existing canopies instead of the points.
   * Because of this it does not need to actually store the points, instead storing a total points vector and the number
   * of points. From this a centroid can be computed. <p/> This method is used by the CanopyReducer.
   *
   * @param point    the point to be added
   * @param canopies the List<Canopy> to be appended
   */
  public static void addPointToCanopies(Vector point, List<Canopy> canopies) {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < t1) {
        canopy.addPoint(point);
      }
      pointStronglyBound = pointStronglyBound || (dist < t2);
    }
    if (!pointStronglyBound) {
      canopies.add(new Canopy(point));
    }
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests and to emit the point and its covering
   * canopies to the output. The CanopyCombiner will then sum the canopy points and produce the centroids.
   *
   * @param point     the point to be added
   * @param canopies  the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public static void emitPointToNewCanopies(Vector point,
                                            List<Canopy> canopies, OutputCollector<Text, Vector> collector)
      throws IOException {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < t1) {
        canopy.emitPoint(point, collector);
      }
      pointStronglyBound = pointStronglyBound || (dist < t2);
    }
    if (!pointStronglyBound) {
      Canopy canopy = new Canopy(point);
      canopies.add(canopy);
      canopy.emitPoint(point, collector);
    }
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests and to emit the point keyed by its
   * covering canopies to the output. if the point is not covered by any canopies (due to canopy centroid clustering),
   * emit the point to the closest covering canopy.
   *
   * @param point     the point to be added
   * @param canopies  the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public static void emitPointToExistingCanopies(Vector point,
                                                 List<Canopy> canopies,
                                                 OutputCollector<Text, Vector> collector) throws IOException {
    double minDist = Double.MAX_VALUE;
    Canopy closest = null;
    boolean isCovered = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < t1) {
        isCovered = true;
        collector.collect(new Text(canopy.getIdentifier()), point);
      } else if (dist < minDist) {
        minDist = dist;
        closest = canopy;
      }
    }
    // if the point is not contained in any canopies (due to canopy centroid
    // clustering), emit the point to the closest covering canopy.
    if (!isCovered) {
      collector.collect(new Text(closest.getIdentifier()), point);
    }
  }


  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    AbstractVector.writeVector(out, computeCentroid());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    this.center = AbstractVector.readVector(in);
    this.pointTotal = center.clone();
    this.numPoints = 1;
  }

  /** Format the canopy for output */
  public static String formatCanopy(Canopy canopy) {
    return "C" + canopy.id + ": "
        + canopy.computeCentroid().asFormatString();
  }

  @Override
  public String asFormatString() {
    return formatCanopy(this);
  }

  /**
   * Decodes and returns a Canopy from the formattedString
   *
   * @param formattedString a String prouced by formatCanopy
   * @return a new Canopy
   */
  public static Canopy decodeCanopy(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    String id = formattedString.substring(0, beginIndex);
    String centroid = formattedString.substring(beginIndex);
    if (id.charAt(0) == 'C') {
      int canopyId = Integer.parseInt(formattedString.substring(1,
          beginIndex - 2));
      Vector canopyCentroid = AbstractVector.decodeVector(centroid);
      return new Canopy(canopyCentroid, canopyId);
    }
    return null;
  }

  /**
   * Add a point to the canopy
   *
   * @param point some point to add
   */
  public void addPoint(Vector point) {
    numPoints++;
    pointTotal = pointTotal.plus(point);

  }

  /**
   * Emit the point to the collector, keyed by the canopy's formatted representation
   *
   * @param point a point to emit.
   */
  public void emitPoint(Vector point, OutputCollector<Text, Vector> collector)
      throws IOException {
    collector.collect(new Text(this.getIdentifier()), point);
  }

  @Override
  public String toString() {
    return getIdentifier() + " - " + center.asFormatString();
  }

  public String getIdentifier() {
    return "C" + id;
  }


  /**
   * Compute the centroid by averaging the pointTotals
   *
   * @return a SparseVector (required by Mapper) which is the new centroid
   */
  public Vector computeCentroid() {
    return pointTotal.divide(numPoints);
  }

  /**
   * Return if the point is covered by this canopy
   *
   * @param point a point
   * @return if the point is covered
   */
  public boolean covers(Vector point) {
    return measure.distance(center.getLengthSquared(), center, point) < t1;
  }
}
