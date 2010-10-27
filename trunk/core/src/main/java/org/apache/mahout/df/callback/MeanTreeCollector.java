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

package org.apache.mahout.df.callback;

import org.apache.mahout.df.data.Data;

/**
 * Computes the error rate for each tree, and returns the mean of all the trees
 */
public class MeanTreeCollector implements PredictionCallback {
  
  /** number of errors for each tree */
  private final int[] nbErrors;
  
  /** number of predictions for each tree */
  private final int[] nbPredictions;
  
  private final Data data;
  
  public MeanTreeCollector(Data data, int nbtrees) {
    nbErrors = new int[nbtrees];
    nbPredictions = new int[nbtrees];
    this.data = data;
  }
  
  public double meanTreeError() {
    double sumerror = 0.0;
    
    for (int treeId = 0; treeId < nbErrors.length; treeId++) {
      if (nbPredictions[treeId] == 0) {
        continue; // this tree has 0 predictions
      }
      
      sumerror += (double) nbErrors[treeId] / nbPredictions[treeId];
    }
    
    return sumerror / nbErrors.length;
  }
  
  @Override
  public void prediction(int treeId, int instanceId, int prediction) {
    if (prediction == -1) {
      return;
    }
    
    nbPredictions[treeId]++;
    
    if (data.get(instanceId).getLabel() != prediction) {
      nbErrors[treeId]++;
    }
  }
  
}
