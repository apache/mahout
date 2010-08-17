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
import java.io.IOException;

import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according
 * to the application of some distance metric, and a point total which is the sum of all the points and is
 * used to compute the centroid when needed.
 */
public class Canopy extends AbstractCluster {
  
  /** Used for deserializaztion as a writable */
  public Canopy() { }
  
  /**
   * Create a new Canopy containing the given point and canopyId
   * 
   * @param center
   *          a point in vector space
   * @param canopyId
   *          an int identifying the canopy local to this process only
   */
  public Canopy(Vector center, int canopyId) {
    this.setId(canopyId);
    this.setNumPoints(0);
    this.setCenter(new RandomAccessSparseVector(center));
    this.setRadius(center.like());
    observe(center);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
  }
  
  /** Format the canopy for output */
  public static String formatCanopy(Canopy canopy) {
    return "C" + canopy.getId() + ": " + canopy.computeCentroid().asFormatString();
  }
  
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
  
  @Override
  public String toString() {
    return getIdentifier() + ": " + getCenter().asFormatString();
  }
  
  @Override
  public String getIdentifier() {
    return "C-" + getId();
  }
}
