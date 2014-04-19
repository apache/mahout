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
package org.apache.mahout.clustering.iterator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

/**
 * This is a probability-weighted clustering policy, suitable for fuzzy k-means
 * clustering
 * 
 */
public class FuzzyKMeansClusteringPolicy extends AbstractClusteringPolicy {

  private double m = 2;
  private double convergenceDelta = 0.05;

  public FuzzyKMeansClusteringPolicy() {
  }

  public FuzzyKMeansClusteringPolicy(double m, double convergenceDelta) {
    this.m = m;
    this.convergenceDelta = convergenceDelta;
  }

  @Override
  public Vector select(Vector probabilities) {
    return probabilities;
  }
  
  @Override
  public Vector classify(Vector data, ClusterClassifier prior) {
    Collection<SoftCluster> clusters = Lists.newArrayList();
    List<Double> distances = Lists.newArrayList();
    for (Cluster model : prior.getModels()) {
      SoftCluster sc = (SoftCluster) model;
      clusters.add(sc);
      distances.add(sc.getMeasure().distance(data, sc.getCenter()));
    }
    FuzzyKMeansClusterer fuzzyKMeansClusterer = new FuzzyKMeansClusterer();
    fuzzyKMeansClusterer.setM(m);
    return fuzzyKMeansClusterer.computePi(clusters, distances);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(m);
    out.writeDouble(convergenceDelta);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.m = in.readDouble();
    this.convergenceDelta = in.readDouble();
  }

  @Override
  public void close(ClusterClassifier posterior) {
    for (Cluster cluster : posterior.getModels()) {
      ((org.apache.mahout.clustering.kmeans.Kluster) cluster).calculateConvergence(convergenceDelta);
      cluster.computeParameters();
    }
    
  }
  
}
