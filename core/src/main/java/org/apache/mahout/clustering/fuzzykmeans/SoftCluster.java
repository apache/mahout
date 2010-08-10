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

import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.Vector;

public class SoftCluster extends Cluster {

  // For Writable
  public SoftCluster() {
  }

  /**
   * Construct a new SoftCluster with the given point as its center
   *
   * @param center
   *          the center point
   */
  public SoftCluster(Vector center, int clusterId) {
    super(center, clusterId);
  }

  /**
   * Format the SoftCluster for output
   *
   * @param cluster
   *          the Cluster
   */
  public static String formatCluster(SoftCluster cluster) {
    return cluster.getIdentifier() + ": " + cluster.computeCentroid().asFormatString();
  }

  /**
   * Decodes and returns a SoftCluster from the formattedString
   *
   * @param formattedString
   *          a String produced by formatCluster
   */
  public static SoftCluster decodeCluster(String formattedString) {
    int beginIndex = formattedString.indexOf('{');
    String id = formattedString.substring(0, beginIndex);
    String center = formattedString.substring(beginIndex);
    char firstChar = id.charAt(0);
    boolean startsWithV = firstChar == 'V';
    if ((firstChar == 'C') || startsWithV) {
      int clusterId = Integer.parseInt(formattedString.substring(1, beginIndex - 2));
      Vector clusterCenter = AbstractVector.decodeVector(center);

      SoftCluster cluster = new SoftCluster(clusterCenter, clusterId);
      cluster.setConverged(startsWithV);
      return cluster;
    }
    return null;
  }

  @Override
  public String asFormatString() {
    return formatCluster(this);
  }

  @Override
  public String getIdentifier() {
    return (isConverged() ? "SV-" : "SC-") + getId();
  }
}
