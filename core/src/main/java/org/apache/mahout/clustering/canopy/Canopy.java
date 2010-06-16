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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according
 * to the application of some distance metric, and a point total which is the sum of all the points and is
 * used to compute the centroid when needed.
 */
public class Canopy extends ClusterBase {
  
  /** Used for deserializaztion as a writable */
  public Canopy() { }
  
  /**
   * Create a new Canopy containing the given point and canopyId
   * 
   * @param point
   *          a point in vector space
   * @param canopyId
   *          an int identifying the canopy local to this process only
   */
  public Canopy(Vector point, int canopyId) {
    this.setId(canopyId);
    this.setCenter(new RandomAccessSparseVector(point));
    this.setPointTotal(getCenter().clone());
    this.setNumPoints(1);
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    super.write(out);
    VectorWritable.writeVector(out, computeCentroid());
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    VectorWritable temp = new VectorWritable();
    temp.readFields(in);
    this.setCenter(new RandomAccessSparseVector(temp.get()));
    this.setPointTotal(getCenter().clone());
    this.setNumPoints(1);
  }
  
  /** Format the canopy for output */
  public static String formatCanopy(Canopy canopy) {
    return "C" + canopy.getId() + ": " + canopy.computeCentroid().asFormatString();
  }
  
  @Override
  public String asFormatString() {
    return formatCanopy(this);
  }
  
  /**
   * Decodes and returns a Canopy from the formattedString
   * 
   * @param formattedString
   *          a String prouced by formatCanopy
   * @return a new Canopy
   */
  public static Canopy decodeCanopy(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    String id = formattedString.substring(0, beginIndex);
    String centroid = formattedString.substring(beginIndex);
    if (id.charAt(0) == 'C') {
      int canopyId = Integer.parseInt(formattedString.substring(1, beginIndex - 2));
      Vector canopyCentroid = AbstractVector.decodeVector(centroid);
      return new Canopy(canopyCentroid, canopyId);
    }
    return null;
  }
  
  /**
   * Add a point to the canopy
   * 
   * @param point
   *          some point to add
   */
  public void addPoint(Vector point) {
    setNumPoints(getNumPoints() + 1);
    point.addTo(getPointTotal());
  }
  
  @Override
  public String toString() {
    return getIdentifier() + ": " + getCenter().asFormatString();
  }
  
  @Override
  public String getIdentifier() {
    return "C-" + getId();
  }
  
  /**
   * Compute the centroid by averaging the pointTotals
   * 
   * @return a RandomAccessSparseVector (required by Mapper) which is the new centroid
   */
  @Override
  public Vector computeCentroid() {
    return getPointTotal().divide(getNumPoints());
  }
}
