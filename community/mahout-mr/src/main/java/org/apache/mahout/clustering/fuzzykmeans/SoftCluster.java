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

import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class SoftCluster extends Kluster {
  
  // For Writable
  public SoftCluster() {}
  
  /**
   * Construct a new SoftCluster with the given point as its center
   * 
   * @param center
   *          the center point
   * @param measure
   *          the DistanceMeasure
   */
  public SoftCluster(Vector center, int clusterId, DistanceMeasure measure) {
    super(center, clusterId, measure);
  }
  
  @Override
  public String asFormatString() {
    return this.getIdentifier() + ": "
        + this.computeCentroid().asFormatString();
  }
  
  @Override
  public String getIdentifier() {
    return (isConverged() ? "SV-" : "SC-") + getId();
  }
  
  @Override
  public double pdf(VectorWritable vw) {
    // SoftCluster pdf cannot be calculated out of context. See
    // FuzzyKMeansClusterer
    throw new UnsupportedOperationException(
        "SoftCluster pdf cannot be calculated out of context. See FuzzyKMeansClusterer");
  }
}
