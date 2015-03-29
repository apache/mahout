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

import java.util.Collection;
import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class FuzzyKMeansClusterer {

  private static final double MINIMAL_VALUE = 0.0000000001;
  
  private double m = 2.0; // default value
  
  public Vector computePi(Collection<SoftCluster> clusters, List<Double> clusterDistanceList) {
    Vector pi = new DenseVector(clusters.size());
    for (int i = 0; i < clusters.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);
      pi.set(i, probWeight);
    }
    return pi;
  }
  
  /** Computes the probability of a point belonging to a cluster */
  public double computeProbWeight(double clusterDistance, Iterable<Double> clusterDistanceList) {
    if (clusterDistance == 0) {
      clusterDistance = MINIMAL_VALUE;
    }
    double denom = 0.0;
    for (double eachCDist : clusterDistanceList) {
      if (eachCDist == 0.0) {
        eachCDist = MINIMAL_VALUE;
      }
      denom += Math.pow(clusterDistance / eachCDist, 2.0 / (m - 1));
    }
    return 1.0 / denom;
  }

  public void setM(double m) {
    this.m = m;
  }
}
