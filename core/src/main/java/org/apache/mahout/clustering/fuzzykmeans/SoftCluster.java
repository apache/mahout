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

package org.apache.mahout.clustering.fuzzykmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SquareRootFunction;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class SoftCluster implements Writable {

  // this cluster's clusterId
  private int clusterId;

  // the current center
  private Vector center = new RandomAccessSparseVector(0);

  // the current centroid is lazy evaluated and may be null
  private Vector centroid = null;

  // The Probability of belongingness sum
  private double pointProbSum = 0.0;

  // the total of all points added to the cluster
  private Vector weightedPointTotal = null;

  // has the centroid converged with the center?
  private boolean converged = false;

  // track membership parameters
  private double s0 = 0;

  private Vector s1;

  private Vector s2;
  
  /**
   * Format the SoftCluster for output
   *
   * @param cluster the Cluster
   */
  public static String formatCluster(SoftCluster cluster) {
    return cluster.getIdentifier() + ": "
        + cluster.computeCentroid().asFormatString();
  }

  /**
   * Decodes and returns a SoftCluster from the formattedString
   *
   * @param formattedString a String produced by formatCluster
   */
  public static SoftCluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    if (firstChar == 'C' || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1, beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);

      SoftCluster cluster = new SoftCluster(clusterCenter, clusterId);
      cluster.setConverged(startsWithV);
      return cluster;
    }
    return null;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(clusterId);
    out.writeBoolean(converged);
    Vector vector = computeCentroid();
    VectorWritable.writeVector(out, vector);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    clusterId = in.readInt();
    converged = in.readBoolean();
    center = VectorWritable.readVector(in);
    this.pointProbSum = 0;
    this.weightedPointTotal = center.like();
  }

  /**
   * Compute the centroid
   *
   * @return the new centroid
   */
  public Vector computeCentroid() {
    if (pointProbSum == 0) {
      return weightedPointTotal;
    } else if (centroid == null) {
      // lazy compute new centroid
      centroid = weightedPointTotal.divide(pointProbSum);
    }
    return centroid;
  }

  //For Writable
  public SoftCluster() {
  }

  /**
   * Construct a new SoftCluster with the given point as its center
   *
   * @param center the center point
   */
  public SoftCluster(Vector center) {
    this.center = center;
    this.pointProbSum = 0;

    this.weightedPointTotal = center.like();
  }

  /**
   * Construct a new SoftCluster with the given point as its center
   *
   * @param center the center point
   */
  public SoftCluster(Vector center, int clusterId) {
    this.clusterId = clusterId;
    this.center = center;
    this.pointProbSum = 0;
    this.weightedPointTotal = center.like();
  }

  /** Construct a new softcluster with the given clusterID */
  public SoftCluster(String clusterId) {

    this.clusterId = Integer.parseInt((clusterId.substring(1)));
    this.pointProbSum = 0;
    // this.weightedPointTotal = center.like();
    this.converged = clusterId.charAt(0) == 'V';
  }

  @Override
  public String toString() {
    return getIdentifier() + " - " + center.asFormatString();
  }

  public String getIdentifier() {
    if (converged) {
      return "V" + clusterId;
    } else {
      return "C" + clusterId;
    }
  }

  /** Observe the point, accumulating weighted variables for std() calculation */
  private void observePoint(Vector point, double ptProb) {
    s0 += ptProb;
    Vector wtPt = point.times(ptProb);
    if (s1 == null) {
      s1 = point.clone();
    } else {
      s1 = s1.plus(wtPt);
    }
    if (s2 == null) {
      s2 = wtPt.times(wtPt);
    } else {
      s2 = s2.plus(wtPt.times(wtPt));
    }
  }

  /** Compute a "standard deviation" value to use as the "radius" of the cluster for display purposes */
  public double std() {
    if (s0 > 0) {
      Vector radical = s2.times(s0).minus(s1.times(s1));
      radical = radical.times(radical).assign(new SquareRootFunction());
      Vector stds = radical.assign(new SquareRootFunction()).divide(s0);
      return stds.zSum() / stds.size();
    } else {
      return 0;
    }
  }

  /**
   * Add the point to the SoftCluster
   *
   * @param point a point to add
   */
  public void addPoint(Vector point, double ptProb) {
    observePoint(point, ptProb);
    centroid = null;
    pointProbSum += ptProb;
    if (weightedPointTotal == null) {
      weightedPointTotal = point.clone().times(ptProb);
    } else {
      weightedPointTotal = weightedPointTotal.plus(point.times(ptProb));
    }
  }

  /**
   * Add the point to the cluster
   *
   * @param delta a point to add
   */
  public void addPoints(Vector delta, double partialSumPtProb) {
    centroid = null;
    pointProbSum += partialSumPtProb;
    if (weightedPointTotal == null) {
      weightedPointTotal = delta.clone();
    } else {
      weightedPointTotal = weightedPointTotal.plus(delta);
    }
  }

  public Vector getCenter() {
    return center;
  }

  public double getPointProbSum() {
    return pointProbSum;
  }

  /** Compute the centroid and set the center to it. */
  public void recomputeCenter() {
    center = computeCentroid();
    pointProbSum = 0;
    weightedPointTotal = center.like();
  }

  public Vector getWeightedPointTotal() {
    return weightedPointTotal;
  }

  public void setWeightedPointTotal(Vector v) {
    this.weightedPointTotal = v;
  }

  public boolean isConverged() {
    return converged;
  }

  public void setConverged(boolean converged) {
    this.converged = converged;
  }

  public int getClusterId() {
    return clusterId;
  }



}
