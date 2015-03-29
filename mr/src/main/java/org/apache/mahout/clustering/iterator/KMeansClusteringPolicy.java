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

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;

/**
 * This is a simple maximum likelihood clustering policy, suitable for k-means
 * clustering
 * 
 */
public class KMeansClusteringPolicy extends AbstractClusteringPolicy {
  
  public KMeansClusteringPolicy() {
  }
  
  public KMeansClusteringPolicy(double convergenceDelta) {
    this.convergenceDelta = convergenceDelta;
  }
  
  private double convergenceDelta = 0.001;

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(convergenceDelta);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.convergenceDelta = in.readDouble();
  }
  
  @Override
  public void close(ClusterClassifier posterior) {
    boolean allConverged = true;
    for (Cluster cluster : posterior.getModels()) {
      org.apache.mahout.clustering.kmeans.Kluster kluster = (org.apache.mahout.clustering.kmeans.Kluster) cluster;
      boolean converged = kluster.calculateConvergence(convergenceDelta);
      allConverged = allConverged && converged;
      cluster.computeParameters();
    }
    
  }
  
}
