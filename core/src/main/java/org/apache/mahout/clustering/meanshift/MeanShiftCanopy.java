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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.list.IntArrayList;

/**
 * This class models a canopy as a center point, the number of points that are
 * contained within it according to the application of some distance metric, and
 * a point total which is the sum of all the points and is used to compute the
 * centroid when needed.
 */
public class MeanShiftCanopy extends Cluster {
  
  // TODO: this is still problematic from a scalability perspective, but how
  // else to encode membership?
  private IntArrayList boundPoints = new IntArrayList();
  
  /**
   * Used for Writable
   */
  public MeanShiftCanopy() {}
  
  /**
   * Create a new Canopy containing the given point
   * 
   * @param point
   *          a Vector
   * @param id
   *          an int canopy id
   * @param measure
   *          a DistanceMeasure
   */
  public MeanShiftCanopy(Vector point, int id, DistanceMeasure measure) {
    super(point, id, measure);
    boundPoints.add(id);
  }
  
  /**
   * Create an initial Canopy, retaining the original type of the given point
   * (e.g. NamedVector)
   * 
   * @param point
   *          a Vector
   * @param id
   *          an int
   * @param measure
   *          a DistanceMeasure
   * @return a MeanShiftCanopy
   */
  public static MeanShiftCanopy initialCanopy(Vector point, int id,
      DistanceMeasure measure) {
    MeanShiftCanopy result = new MeanShiftCanopy(point, id, measure);
    // overwrite center so original point type is retained
    result.setCenter(point);
    return result;
  }
  
  /**
   * Create a new Canopy containing the given point, id and bound points
   * 
   * @param point
   *          a Vector
   * @param id
   *          an int identifying the canopy local to this process only
   * @param boundPoints
   *          a IntArrayList containing points ids bound to the canopy
   * @param converged
   *          true if the canopy has converged
   */
  MeanShiftCanopy(Vector point, int id, IntArrayList boundPoints,
      boolean converged) {
    this.setId(id);
    this.setCenter(point);
    this.setRadius(point.like());
    this.setNumPoints(1);
    this.boundPoints = boundPoints;
    setConverged(converged);
  }
  
  public IntArrayList getBoundPoints() {
    return boundPoints;
  }
  
  /**
   * The receiver overlaps the given canopy. Add my bound points to it.
   * 
   * @param canopy
   *          an existing MeanShiftCanopy
   */
  void merge(MeanShiftCanopy canopy) {
    boundPoints.addAllOf(canopy.boundPoints);
  }
  
  /**
   * The receiver touches the given canopy. Add respective centers with the
   * given weights.
   * 
   * @param canopy
   *          an existing MeanShiftCanopy
   * @param weight
   *          double weight of the touching
   */
  void touch(MeanShiftCanopy canopy, double weight) {
    canopy.observe(getCenter(), weight * boundPoints.size());
    observe(canopy.getCenter(), weight * canopy.boundPoints.size());
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    int numpoints = in.readInt();
    this.boundPoints = new IntArrayList();
    for (int i = 0; i < numpoints; i++) {
      this.boundPoints.add(in.readInt());
    }
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeInt(boundPoints.size());
    for (int v : boundPoints.elements()) {
      out.writeInt(v);
    }
  }
  
  public MeanShiftCanopy shallowCopy() {
    MeanShiftCanopy result = new MeanShiftCanopy();
    result.setMeasure(this.getMeasure());
    result.setId(this.getId());
    result.setCenter(this.getCenter());
    result.setRadius(this.getRadius());
    result.setNumPoints(this.getNumPoints());
    result.setBoundPoints(boundPoints);
    return result;
  }
  
  @Override
  public String asFormatString() {
    return toString();
  }
  
  public void setBoundPoints(IntArrayList boundPoints) {
    this.boundPoints = boundPoints;
  }
  
  @Override
  public String getIdentifier() {
    return (isConverged() ? "MSV-" : "MSC-") + getId();
  }
  
  @Override
  public double pdf(VectorWritable vw) {
    // MSCanopy membership is explicit via membership in boundPoints. Can't
    // compute pdf for Arbitrary point
    throw new UnsupportedOperationException();
  }
  
}
