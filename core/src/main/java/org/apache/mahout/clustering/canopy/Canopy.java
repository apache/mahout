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

import org.apache.mahout.clustering.iterator.DistanceMeasureCluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;

/**
 * This class models a canopy as a center point, the number of points that are contained within it according
 * to the application of some distance metric, and a point total which is the sum of all the points and is
 * used to compute the centroid when needed.
 */
public class Canopy extends DistanceMeasureCluster {
  
  /** Used for deserialization as a writable */
  public Canopy() { }
  
  /**
   * Create a new Canopy containing the given point and canopyId
   * 
   * @param center a point in vector space
   * @param canopyId an int identifying the canopy local to this process only
   * @param measure a DistanceMeasure to use
   */
  public Canopy(Vector center, int canopyId, DistanceMeasure measure) {
    super(center, canopyId, measure);
    observe(center);
  }

  public String asFormatString() {
    return "C" + this.getId() + ": " + this.computeCentroid().asFormatString();
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
