package org.apache.mahout.clustering.dirichlet;

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

import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

public class DirichletState<O> {

  public int numClusters; // the number of clusters

  public ModelDistribution<O> modelFactory; // the factory for models

  public List<DirichletCluster<O>> clusters; // the clusters for this iteration

  public Vector mixture; // the mixture vector

  public double offset; // alpha_0 / numClusters

  @SuppressWarnings("unchecked")
  public DirichletState(ModelDistribution<O> modelFactory,
                        int numClusters, double alpha_0, int thin, int burnin) {
    this.numClusters = numClusters;
    this.modelFactory = modelFactory;
    // initialize totalCounts
    offset = alpha_0 / numClusters;
    // sample initial prior models
    clusters = new ArrayList<DirichletCluster<O>>();
    for (Model<?> m : modelFactory.sampleFromPrior(numClusters)) {
      clusters.add(new DirichletCluster(m, offset));
    }
    // sample the mixture parameters from a Dirichlet distribution on the totalCounts 
    mixture = UncommonDistributions.rDirichlet(totalCounts());
  }

  public DirichletState() {
  }

  public Vector totalCounts() {
    Vector result = new DenseVector(numClusters);
    for (int i = 0; i < numClusters; i++) {
      result.set(i, clusters.get(i).totalCount);
    }
    return result;
  }

  /**
   * Update the receiver with the new models
   *
   * @param newModels a Model<Observation>[] of new models
   */
  public void update(Model<O>[] newModels) {
    // compute new model parameters based upon observations and update models
    for (int i = 0; i < newModels.length; i++) {
      newModels[i].computeParameters();
      clusters.get(i).setModel(newModels[i]);
    }
    // update the mixture
    mixture = UncommonDistributions.rDirichlet(totalCounts());
  }

  /**
   * return the adjusted probability that x is described by the kth model
   *
   * @param x an Observation
   * @param k an int index of a model
   * @return the double probability
   */
  public double adjustedProbability(O x, int k) {
    double pdf = clusters.get(k).model.pdf(x);
    double mix = mixture.get(k);
    return mix * pdf;
  }

  @SuppressWarnings("unchecked")
  public Model<O>[] getModels() {
    Model<O>[] result = new Model[numClusters];
    for (int i = 0; i < numClusters; i++) {
      result[i] = clusters.get(i).model;
    }
    return result;
  }

}
