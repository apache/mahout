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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

public class Cluster extends AbstractCluster {

  /** Error message for unknown cluster format in output. */
  private static final String ERROR_UNKNOWN_CLUSTER_FORMAT = "Unknown cluster format:\n";

  /** Has the centroid converged with the center? */
  private boolean converged;

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
    this.setId(clusterId);
    this.setNumPoints(0);
    this.setCenter(new RandomAccessSparseVector(center));
    this.setRadius(center.like());
  }

  /**
   * Format the cluster for output
   * 
   * @param cluster
   *          the Cluster
   * @return the String representation of the Cluster
   */
  public static String formatCluster(Cluster cluster) {
    return cluster.getIdentifier() + ": " + cluster.computeCentroid().asFormatString();
  }

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
  public static AbstractCluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    if (beginIndex <= 0) {
      throw new IllegalArgumentException(ERROR_UNKNOWN_CLUSTER_FORMAT + formattedString);
    }
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    Cluster cluster;
    if ((firstChar == 'C') || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1, beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);
      cluster = new Cluster(clusterCenter, clusterId);
      cluster.setConverged(startsWithV);
    } else {
      throw new IllegalArgumentException(ERROR_UNKNOWN_CLUSTER_FORMAT + formattedString);
    }
    return cluster;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeBoolean(converged);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    this.converged = in.readBoolean();
  }

  @Override
  public String toString() {
    return asFormatString(null);
  }

  @Override
  @Override
  public String getIdentifier() {
    return (converged ? "VL-" : "CL-") + getId();
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
  public boolean computeConvergence(DistanceMeasure measure, double convergenceDelta) {
    Vector centroid = computeCentroid();
    converged = measure.distance(centroid.getLengthSquared(), centroid, getCenter()) <= convergenceDelta;
    return converged;
  }

  public boolean isConverged() {
    return converged;
  }

  protected void setConverged(boolean converged) {
    this.converged = converged;
  }

}
