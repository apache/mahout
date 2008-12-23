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

package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.DistanceMeasure;

import java.io.IOException;
import java.util.List;

public class Cluster {

  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.kmeans.measure";

  public static final String CLUSTER_PATH_KEY = "org.apache.mahout.clustering.kmeans.path";

  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.kmeans.convergence";

  private static int nextClusterId = 0;

  // this cluster's clusterId
  private final int clusterId;

  // the current center
  private Vector center = new SparseVector(0);

  // the current centroid is lazy evaluated and may be null
  private Vector centroid = null;

  // the number of points in the cluster
  private int numPoints = 0;

  // the total of all points added to the cluster
  private Vector pointTotal = null;

  // has the centroid converged with the center?
  private boolean converged = false;

  private static DistanceMeasure measure;

  private static double convergenceDelta = 0;

  /**
   * Format the cluster for output
   * 
   * @param cluster
   *            the Cluster
   * @return
   */
  public static String formatCluster(Cluster cluster) {
    return cluster.getIdentifier() + ": "
        + cluster.computeCentroid().asFormatString();
  }

  /**
   * Decodes and returns a Cluster from the formattedString
   * 
   * @param formattedString
   *            a String produced by formatCluster
   * @return a new Canopy
   */
  public static Cluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('[');
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    if (firstChar == 'C' || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1, beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);
      Cluster cluster = new Cluster(clusterCenter, clusterId);
      cluster.converged = startsWithV;
      return cluster;
    }
    return null;
  }

  /**
   * Configure the distance measure from the job
   * 
   * @param job
   *            the JobConf for the job
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
   * @param aMeasure
   *            the DistanceMeasure
   * @param aConvergenceDelta
   *            the delta value used to define convergence
   */
  public static void config(DistanceMeasure aMeasure, double aConvergenceDelta) {
    measure = aMeasure;
    convergenceDelta = aConvergenceDelta;
    nextClusterId = 0;
  }

  /**
   * Emit the point to the nearest cluster center
   * 
   * @param point
   *            a point
   * @param clusters
   *            a List<Cluster> to test
   * @param values
   *            a Writable containing the input point and possible other values
   *            of interest (payload)
   * @param output
   *            the OutputCollector to emit into
   * @throws IOException
   */
  public static void emitPointToNearestCluster(Vector point,
      List<Cluster> clusters, Text values, OutputCollector<Text, Text> output)
      throws IOException {
    Cluster nearestCluster = null;
    double nearestDistance = Double.MAX_VALUE;
    for (Cluster cluster : clusters) {
      double distance = measure.distance(point, cluster.getCenter());
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
   * @return the new centroid
   */
  private Vector computeCentroid() {
    if (numPoints == 0)
      return pointTotal;
    else if (centroid == null) {
      // lazy compute new centroid
      centroid = pointTotal.divide(numPoints);
    }
    return centroid;
  }

  /**
   * Construct a new cluster with the given point as its center
   * 
   * @param center
   *            the center point
   */
  public Cluster(Vector center) {
    this.clusterId = nextClusterId++;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = center.like();
  }

  /**
   * Construct a new cluster with the given point as its center
   * 
   * @param center
   *            the center point
   */
  public Cluster(Vector center, int clusterId) {
    this.clusterId = clusterId;
    this.center = center;
    this.numPoints = 0;
    this.pointTotal = center.like();
  }

  @Override
  public String toString() {
    return getIdentifier() + " - " + center.asFormatString();
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
   * @param point
   *            a point to add
   */
  public void addPoint(Vector point) {
    centroid = null;
    numPoints++;
    if (pointTotal == null)
      pointTotal = point.copy();
    else
      pointTotal = point.plus(pointTotal);
  }

  /**
   * Add the point to the cluster
   * 
   * @param count
   *            the number of points in the delta
   * @param delta
   *            a point to add
   */
  public void addPoints(int count, Vector delta) {
    centroid = null;
    numPoints += count;
    if (pointTotal == null)
      pointTotal = delta.copy();
    else
      pointTotal = delta.plus(pointTotal);
  }

  public Vector getCenter() {
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
    pointTotal = center.like();
  }

  /**
   * Return if the cluster is converged by comparing its center and centroid.
   * 
   * @return if the cluster is converged
   */
  public boolean computeConvergence() {
    Vector centroid = computeCentroid();
    converged = measure.distance(centroid, center) <= convergenceDelta;
    return converged;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  public boolean isConverged() {
    return converged;
  }

}
