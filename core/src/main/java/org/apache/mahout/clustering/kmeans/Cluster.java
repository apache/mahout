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
package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.SquareRootFunction;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

public class Cluster extends ClusterBase implements Writable {

  private static final String ERROR_UNKNOWN_CLUSTER_FORMAT = "Unknown cluster format:\n";

  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.kmeans.measure";

  public static final String CLUSTER_PATH_KEY = "org.apache.mahout.clustering.kmeans.path";

  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.kmeans.convergence";

  /** The number of iterations that have taken place */
  public static final String ITERATION_NUMBER = "org.apache.mahout.clustering.kmeans.iteration";
  /** Boolean value indicating whether the initial input is from Canopy clustering */
  public static final String CANOPY_INPUT = "org.apache.mahout.clustering.kmeans.canopyInput";

  private static int nextClusterId = 0;


  // the current centroid is lazy evaluated and may be null
  private Vector centroid = null;


  // the total of all the points squared, used for std computation
  private Vector pointSquaredTotal = null;

  // has the centroid converged with the center?
  private boolean converged = false;
  private static DistanceMeasure measure;
  private static double convergenceDelta = 0;

  /**
   * Format the cluster for output
   *
   * @param cluster the Cluster
   * @return the String representation of the Cluster
   */
  public static String formatCluster(Cluster cluster) {
    return cluster.getIdentifier() + ": "
        + cluster.computeCentroid().asFormatString();
  }

  @Override
  public String asFormatString() {
    return formatCluster(this);
  }

  /**
   * Decodes and returns a Cluster from the formattedString
   *
   * @param formattedString a String produced by formatCluster
   * @return a decoded Cluster, not null
   * @throws IllegalArgumentException when the string is wrongly formatted
   */
  public static Cluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    if (firstChar == 'C' || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1,
          beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);
      Cluster cluster = new Cluster(clusterCenter, clusterId);
      cluster.converged = startsWithV;
      return cluster;
    } else {
      throw new IllegalArgumentException(ERROR_UNKNOWN_CLUSTER_FORMAT
          + formattedString);
    }
  }


  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeBoolean(converged);
    AbstractVector.writeVector(out, computeCentroid());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    this.converged = in.readBoolean();
    this.center = AbstractVector.readVector(in);
    this.numPoints = 0;
    this.pointTotal = center.like();
    this.pointSquaredTotal = center.like();
  }

  /**
   * Configure the distance measure from the job
   *
   * @param job the JobConf for the job
   */
  public static void configure(JobConf job) {
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job.get(DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);
      convergenceDelta = Double.parseDouble(job.get(CLUSTER_CONVERGENCE_KEY));
      nextClusterId = 0;
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Configure the distance measure directly. Used by unit tests.
   *
   * @param aMeasure          the DistanceMeasure
   * @param aConvergenceDelta the delta value used to define convergence
   */
  public static void config(DistanceMeasure aMeasure, double aConvergenceDelta) {
    measure = aMeasure;
    convergenceDelta = aConvergenceDelta;
    nextClusterId = 0;
  }

  /**
   * Emit the point to the nearest cluster center
   *
   * @param point    a point
   * @param clusters a List<Cluster> to test
   * @param output   the OutputCollector to emit into
   */
  public static void emitPointToNearestCluster(Vector point,
                                               List<Cluster> clusters, OutputCollector<Text, KMeansInfo> output)
      throws IOException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = measure.distance(clusterCenter.getLengthSquared(), clusterCenter, point);
      if (distance < nearestDistance || nearestCluster == null ) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    // emit only clusterID
    output.collect(new Text(nearestCluster.getIdentifier()), new KMeansInfo(1, point));
  }

  public static void outputPointWithClusterInfo(Vector point,
                                                List<Cluster> clusters, OutputCollector<Text, Text> output)
      throws IOException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      Vector clusterCenter = cluster.getCenter();
      double distance = measure.distance(clusterCenter.getLengthSquared(), clusterCenter, point);
      if (distance < nearestDistance || nearestCluster == null) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    //TODO: this is ugly
    String name = point.getName();
    output.collect(new Text(name != null && name.length() != 0 ? name : point.asFormatString()), new Text(String.valueOf(nearestCluster.id)));
  }

  /**
   * Compute the centroid by averaging the pointTotals
   *
   * @return the new centroid
   */
  private Vector computeCentroid() {
    if (numPoints == 0) {
      return center;
    } else if (centroid == null) {
      // lazy compute new centroid
      centroid = pointTotal.divide(numPoints);
    }
    return centroid;
  }

  /**
   * Construct a new cluster with the given point as its center
   *
   * @param center the center point
   */
  public Cluster(Vector center) {
    super();
    this.id = nextClusterId++;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = center.like();
    this.pointSquaredTotal = center.like();
  }

  /** For (de)serialization as a Writable */
  public Cluster() {
  }

  /**
   * Construct a new cluster with the given point as its center
   *
   * @param center the center point
   */
  public Cluster(Vector center, int clusterId) {
    super();
    this.id = clusterId;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = center.like();
    this.pointSquaredTotal = center.like();
  }

  /** Construct a new clsuter with the given id as identifier */
  public Cluster(String clusterId) {

    this.id = Integer.parseInt((clusterId.substring(1)));
    this.numPoints = 0;
    this.converged = clusterId.startsWith("V");
  }

  @Override
  public String toString() {
    return getIdentifier() + " - " + center.asFormatString();
  }

  public String getIdentifier() {
    if (converged) {
      return "V" + id;
    } else {
      return "C" + id;
    }
  }

  /**
   * Add the point to the cluster
   *
   * @param point a point to add
   */
  public void addPoint(Vector point) {
    addPoints(1, point);
  }

  /**
   * Add the point to the cluster
   *
   * @param count the number of points in the delta
   * @param delta a point to add
   */
  public void addPoints(int count, Vector delta) {
    centroid = null;
    numPoints += count;
    if (pointTotal == null) {
      pointTotal = delta.clone();
      pointSquaredTotal = delta.times(delta);
    } else {
      delta.addTo(pointTotal);
      delta.times(delta).addTo(pointSquaredTotal);
    }
  }


  /** Compute the centroid and set the center to it. */
  public void recomputeCenter() {
    center = computeCentroid();
    numPoints = 0;
    pointTotal = center.like();
  }

  /**
   * Return if the cluster is converged by comparing its center and centroid.
   *
   * @return if the cluster is converged
   */
  public boolean computeConvergence() {
    Vector centroid = computeCentroid();
    converged = measure.distance(centroid.getLengthSquared(), centroid, center) <= convergenceDelta;
    return converged;
  }


  public boolean isConverged() {
    return converged;
  }

  /** @return the std */
  public double getStd() {
    Vector stds = pointSquaredTotal.times(numPoints).minus(
          pointTotal.times(pointTotal)).assign(new SquareRootFunction())
          .divide(numPoints);
    return stds.zSum() / 2;
  }

}
