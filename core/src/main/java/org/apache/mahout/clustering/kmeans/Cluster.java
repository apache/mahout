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

import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.SquareRootFunction;
import org.apache.mahout.math.Vector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Cluster extends ClusterBase {

  /** Error message for unknown cluster format in output. */
  private static final String ERROR_UNKNOWN_CLUSTER_FORMAT = "Unknown cluster format:\n";

  /** The current centroid is lazy evaluated and may be null */
  private Vector centroid = null;

  /** The total of all the points squared, used for std computation */
  private Vector pointSquaredTotal = null;

  /** Has the centroid converged with the center? */
  private boolean converged = false;

  /**
   * Format the cluster for output
   * 
   * @param cluster
   *          the Cluster
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
   * Decodes and returns a Cluster from the formattedString.
   * 
   * @param formattedString
   *          a String produced by formatCluster
   * @return a decoded Cluster, not null
   * @throws IllegalArgumentException
   *           when the string is wrongly formatted
   */
  public static Cluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    if (beginIndex <= 0) {
      throw new IllegalArgumentException(ERROR_UNKNOWN_CLUSTER_FORMAT
          + formattedString);
    }
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    Cluster cluster;
    if (firstChar == 'C' || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1,
          beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);
      cluster = new Cluster(clusterCenter, clusterId);
      cluster.setConverged(startsWithV);
    } else {
      throw new IllegalArgumentException(ERROR_UNKNOWN_CLUSTER_FORMAT
          + formattedString);
    }
    return cluster;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeBoolean(converged);
    VectorWritable.writeVector(out, computeCentroid());
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    this.converged = in.readBoolean();
    this.setCenter(VectorWritable.readVector(in));
    this.setNumPoints(0);
    this.setPointTotal(getCenter().like());
    this.pointSquaredTotal = getCenter().like();
  }

  /**
   * Compute the centroid by averaging the pointTotals
   * 
   * @return the new centroid
   */
  private Vector computeCentroid() {
    if (getNumPoints() == 0) {
      return getCenter();
    } else if (centroid == null) {
      // lazy compute new centroid
      centroid = getPointTotal().divide(getNumPoints());
    }
    return centroid;
  }

  /**
   * Construct a new cluster with the given point as its center
   * 
   * @param center
   *          the center point
   */
  public Cluster(Vector center) {
    super();
    this.setCenter(center);
    this.setNumPoints(0);
    this.setPointTotal(center.like());
    this.pointSquaredTotal = center.like();
  }

  /** For (de)serialization as a Writable */
  public Cluster() {
  }

  /**
   * Construct a new cluster with the given point as its center
   * 
   * @param center
   *          the center point
   */
  public Cluster(Vector center, int clusterId) {
    super();
    this.setId(clusterId);
    this.setCenter(center);
    this.setNumPoints(0);
    this.setPointTotal(center.like());
    this.pointSquaredTotal = center.like();
  }

  /** Construct a new clsuter with the given id as identifier */
  public Cluster(String clusterId) {

    this.setId(Integer.parseInt((clusterId.substring(1))));
    this.setNumPoints(0);
    this.converged = clusterId.startsWith("V");
  }

  @Override
  public String toString() {
    return getIdentifier() + " - " + getCenter().asFormatString();
  }

  public String getIdentifier() {
    if (converged) {
      return "V" + getId();
    } else {
      return "C" + getId();
    }
  }

  /**
   * Add the point to the cluster
   * 
   * @param point
   *          a point to add
   */
  public void addPoint(Vector point) {
    addPoints(1, point);
  }

  /**
   * Add the point to the cluster
   * 
   * @param count
   *          the number of points in the delta
   * @param delta
   *          a point to add
   */
  public void addPoints(int count, Vector delta) {
    centroid = null;
    setNumPoints(getNumPoints() + count);
    if (getPointTotal() == null) {
      setPointTotal(delta.clone());
      pointSquaredTotal = delta.times(delta);
    } else {
      delta.addTo(getPointTotal());
      delta.times(delta).addTo(pointSquaredTotal);
    }
  }

  /** Compute the centroid and set the center to it. */
  public void recomputeCenter() {
    setCenter(computeCentroid());
    setNumPoints(0);
    setPointTotal(getCenter().like());
  }

  /**
   * Return if the cluster is converged by comparing its center and centroid.
   * 
   * @param measure
   *          The distance measure to use for cluster-point comparisons.
   * @param convergenceDelta
   *          the convergence delta to use for stopping.
   * @return if the cluster is converged
   */
  public boolean computeConvergence(DistanceMeasure measure,
      double convergenceDelta) {
    Vector centroid = computeCentroid();
    converged = measure.distance(centroid.getLengthSquared(), centroid,
        getCenter()) <= convergenceDelta;
    return converged;
  }

  public boolean isConverged() {
    return converged;
  }

  private void setConverged(boolean converged) {
    this.converged = converged;
  }

  /** @return the std */
  public double getStd() {
    Vector stds = pointSquaredTotal.times(getNumPoints()).minus(
        getPointTotal().times(getPointTotal()))
        .assign(new SquareRootFunction()).divide(getNumPoints());
    return stds.zSum() / stds.size();
  }

}
