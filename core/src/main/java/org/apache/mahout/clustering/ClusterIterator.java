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
package org.apache.mahout.clustering;

import java.util.List;

import org.apache.mahout.math.Vector;

/**
 * This is an experimental clustering iterator which works with a
 * ClusteringPolicy and a prior ClusterClassifier which has been initialized
 * with a set of models. To date, it has been tested with k-means and Dirichlet
 * clustering. See examples DisplayKMeans and DisplayDirichlet which have been
 * switched over to use it.
 * 
 */
public class ClusterIterator {
  
  public ClusterIterator(ClusteringPolicy policy) {
    super();
    this.policy = policy;
  }
  
  private ClusteringPolicy policy;
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of
   * iterations
   * 
   * @param data
   *          a List<Vector> of input vectors
   * @param prior
   *          the prior-trained ClusterClassifier
   * @param numIterations
   *          the int number of iterations to perform
   * @return the posterior ClusterClassifier
   */
  public ClusterClassifier iterate(List<Vector> data, ClusterClassifier prior,
      int numIterations) {
    for (int iteration = 1; iteration <= numIterations; iteration++) {
      for (Vector vector : data) {
        // classification yields probabilities
        Vector pdfs = prior.classify(vector);
        // policy selects a model given those probabilities
        int selected = policy.select(pdfs);
        // training causes all models to observe data
        prior.train(selected, vector);
      }
      // compute the posterior models
      prior.close();
      // update the policy
      policy.update(prior);
    }
    return prior;
  }
}
