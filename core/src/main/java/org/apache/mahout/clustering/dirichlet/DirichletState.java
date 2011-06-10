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

package org.apache.mahout.clustering.dirichlet;

import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.DistributionDescription;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class DirichletState {
  
  private int numClusters; // the number of clusters
  
  private ModelDistribution<VectorWritable> modelFactory; // the factory for models
  
  private List<DirichletCluster> clusters; // the clusters for this iteration
  
  private Vector mixture; // the mixture vector
  
  private final double alpha0; // alpha0

  public DirichletState(ModelDistribution<VectorWritable> modelFactory,
                        int numClusters,
                        double alpha0) {
    this.numClusters = numClusters;
    this.modelFactory = modelFactory;
    this.alpha0 = alpha0;
    // sample initial prior models
    clusters = Lists.newArrayList();
    for (Model<VectorWritable> m : modelFactory.sampleFromPrior(numClusters)) {
      clusters.add(new DirichletCluster((Cluster) m));
    }
    // sample the mixture parameters from a Dirichlet distribution on the totalCounts
    mixture = UncommonDistributions.rDirichlet(computeTotalCounts(), alpha0);
  }
  
  public DirichletState(DistributionDescription description,
                        int numClusters,
                        double alpha0) {
    this(description.createModelDistribution(), numClusters, alpha0);
  }

  public int getNumClusters() {
    return numClusters;
  }
  
  public void setNumClusters(int numClusters) {
    this.numClusters = numClusters;
  }
  
  public ModelDistribution<VectorWritable> getModelFactory() {
    return modelFactory;
  }
  
  public void setModelFactory(ModelDistribution<VectorWritable> modelFactory) {
    this.modelFactory = modelFactory;
  }
  
  public List<DirichletCluster> getClusters() {
    return clusters;
  }
  
  public void setClusters(List<DirichletCluster> clusters) {
    this.clusters = clusters;
  }
  
  public Vector getMixture() {
    return mixture;
  }
  
  public void setMixture(Vector mixture) {
    this.mixture = mixture;
  }
  
  public Vector totalCounts() {
    return computeTotalCounts();
  }

  private Vector computeTotalCounts() {
    Vector result = new DenseVector(numClusters);
    for (int i = 0; i < numClusters; i++) {
      result.set(i, clusters.get(i).getTotalCount());
    }
    return result;
  }
  
  /**
   * Update the receiver with the new models
   * 
   * @param newModels
   *          a Model[] of new models
   */
  public void update(Cluster[] newModels) {
    // compute new model parameters based upon observations and update models
    for (int i = 0; i < newModels.length; i++) {
      newModels[i].computeParameters();
      clusters.get(i).setModel(newModels[i]);
    }
    // update the mixture
    mixture = UncommonDistributions.rDirichlet(totalCounts(), alpha0);
  }
  
  /**
   * return the adjusted probability that x is described by the kth model
   * 
   * @param x
   *          an Observation
   * @param k
   *          an int index of a model
   * @return the double probability
   */
  public double adjustedProbability(VectorWritable x, int k) {
    double pdf = clusters.get(k).getModel().pdf(x);
    double mix = mixture.get(k);
    return mix * pdf;
  }
  
  public Model<VectorWritable>[] getModels() {
    Model<VectorWritable>[] result = new Model[numClusters];
    for (int i = 0; i < numClusters; i++) {
      result[i] = clusters.get(i).getModel();
    }
    return result;
  }
  
}
