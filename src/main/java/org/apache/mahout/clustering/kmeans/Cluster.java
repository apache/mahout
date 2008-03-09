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
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.utils.DistanceMeasure;
import org.apache.mahout.utils.Point;

import java.io.IOException;
import java.util.List;

public class Cluster {

  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.kmeans.measure";

  public static final String CLUSTER_PATH_KEY = "org.apache.mahout.clustering.kmeans.path";

  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.kmeans.convergence";

  private static int nextClusterId = 0;

  // this cluster's clusterId
  private int clusterId;

  // the current center
  private Float[] center = new Float[0];

  // the current centroid is lazy evaluated and may be null
  private Float[] centroid = null;

  // the number of points in the cluster
  private int numPoints = 0;

  // the total of all points added to the cluster
  private Float[] pointTotal = null;

  // has the centroid converged with the center?
  private boolean converged = false;

  private static DistanceMeasure measure;

  private static float convergenceDelta = 0;

  /**
   * Format the cluster for output
   *
   * @param cluster the Cluster
   * @return
   */
  public static String formatCluster(Cluster cluster) {
    return cluster.getIdentifier() + ": "
            + Point.formatPoint(cluster.computeCentroid());
  }

  /**
   * Decodes and returns a Cluster from the formattedString
   *
   * @param formattedString a String produced by formatCluster
   * @return a new Canopy
   */
  public static Cluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('[');
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    if (id.startsWith("C") || id.startsWith("V")) {
      int clusterId = new Integer(formattedString.substring(1, beginIndex - 2));
      Float[] clusterCenter = Point.decodePoint(center);
      Cluster cluster = new Cluster(clusterCenter, clusterId);
      cluster.converged = id.startsWith("V");
      return cluster;
    }
    return null;
  }

  /**
   * Configure the distance measure from the job
   *
   * @param job the JobConf for the job
   */
  public static void configure(JobConf job) {
    try {
      Class cl = Class.forName(job.get(DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);
      convergenceDelta = new Float(job.get(CLUSTER_CONVERGENCE_KEY));
      nextClusterId = 0;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Configure the distance measure directly. Used by unit tests.
   *
   * @param aMeasure          the DistanceMeasure
   * @param aConvergenceDelta the float delta value used to define convergence
   */
  public static void config(DistanceMeasure aMeasure, float aConvergenceDelta) {
    measure = aMeasure;
    convergenceDelta = aConvergenceDelta;
    nextClusterId = 0;
  }

  /**
   * Emit the point to the nearest cluster center
   *
   * @param point    a Float[] representing the point
   * @param clusters a List<Cluster> to test
   * @param values   a Writable containing the input point and possible other
   *                 values of interest (payload)
   * @param output   the OutputCollector to emit into
   * @throws IOException
   */
  public static void emitPointToNearestCluster(Float[] point,
                                               List<Cluster> clusters, Text values, OutputCollector<Text, Text> output)
          throws IOException {
    Cluster nearestCluster = null;
    float nearestDistance = Float.MAX_VALUE;
    for (Cluster cluster : clusters) {
      float distance = measure.distance(point, cluster.getCenter());
      if (nearestCluster == null || distance < nearestDistance) {
        nearestCluster = cluster;
        nearestDistance = distance;
      }
    }
    output.collect(new Text(formatCluster(nearestCluster)), values);
  }

  /**
   * Compute the centroid by averaging the pointTotals
   *
   * @return a Float[] which is the new centroid
   */
  private Float[] computeCentroid() {
    if (numPoints == 0)
      return pointTotal;
    else if (centroid == null) {
      // lazy compute new centroid
      centroid = new Float[pointTotal.length];
      for (int i = 0; i < pointTotal.length; i++)
        centroid[i] = new Float(pointTotal[i] / numPoints);
    }
    return centroid;
  }

  /**
   * Construct a new cluster with the given point as its center
   *
   * @param center a Float[] center point
   */
  public Cluster(Float[] center) {
    super();
    this.clusterId = nextClusterId++;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = Point.origin(center.length);
  }

  /**
   * Construct a new cluster with the given point as its center
   *
   * @param center a Float[] center point
   */
  public Cluster(Float[] center, int clusterId) {
    super();
    this.clusterId = clusterId;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = Point.origin(center.length);
  }

  /**
   * Return a printable representation of this object, using the user supplied
   * identifier
   *
   * @return
   */
  public String toString() {
    return getIdentifier() + " - " + Point.formatPoint(center);
  }

  public String getIdentifier() {
    if (converged)
      return "V" + clusterId;
    else
      return "C" + clusterId;
  }

  /**
   * Add the point to the cluster
   *
   * @param point a Float[] point to add
   */
  public void addPoint(Float[] point) {
    centroid = null;
    numPoints++;
    if (pointTotal == null)
      pointTotal = point.clone();
    else
      for (int i = 0; i < point.length; i++)
        pointTotal[i] = new Float(point[i] + pointTotal[i]);
  }

  /**
   * Add the point to the cluster
   *
   * @param count the number of points in the delta
   * @param delta a Float[] point to add
   */
  public void addPoints(int count, Float[] delta) {
    centroid = null;
    numPoints += count;
    if (pointTotal == null)
      pointTotal = delta.clone();
    else
      for (int i = 0; i < delta.length; i++)
        pointTotal[i] = new Float(delta[i] + pointTotal[i]);
  }

  public Float[] getCenter() {
    return center;
  }

  public int getNumPoints() {
    return numPoints;
  }

  /**
   * Compute the centroid and set the center to it.
   */
  public void recomputeCenter() {
    center = computeCentroid();
    numPoints = 0;
    pointTotal = Point.origin(center.length);
  }

  /**
   * Return if the cluster is converged by comparing its center and centroid.
   *
   * @return if the cluster is converged
   */
  public boolean computeConvergence() {
    Float[] centroid = computeCentroid();
    converged = measure.distance(centroid, center) <= convergenceDelta;
    return converged;
  }

  public Float[] getPointTotal() {
    return pointTotal;
  }

  public boolean isConverged() {
    return converged;
  }

}
