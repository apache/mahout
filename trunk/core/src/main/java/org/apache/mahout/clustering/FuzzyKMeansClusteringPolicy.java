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

import org.apache.mahout.math.Vector;

/**
 * This is a probability-weighted clustering policy, suitable for fuzzy k-means
 * clustering
 * 
 */
public class FuzzyKMeansClusteringPolicy implements ClusteringPolicy {
    
  /*
   * (non-Javadoc)
   * 
   * @see
   * org.apache.mahout.clustering.ClusteringPolicy#update(org.apache.mahout.
   * clustering.ClusterClassifier)
   */
  @Override
  public void update(ClusterClassifier posterior) {
    // nothing to do here
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.clustering.ClusteringPolicy#select(org.apache.mahout.math.Vector)
   */
  @Override
  public Vector select(Vector probabilities) {
    return probabilities;
  }
  
}
