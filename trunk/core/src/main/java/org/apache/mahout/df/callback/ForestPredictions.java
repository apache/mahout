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

import java.util.Arrays;
import java.util.Random;

import org.apache.mahout.df.data.DataUtils;

/**
 * Collects a forest's predictions
 */
public class ForestPredictions implements PredictionCallback {
  
  /** predictions[n][label] = number of times instance n was classified 'label' */
  private final int[][] predictions;
  
  public ForestPredictions(int nbInstances, int nblabels) {
    predictions = new int[nbInstances][];
    for (int index = 0; index < predictions.length; index++) {
      predictions[index] = new int[nblabels];
    }
  }
  
  @Override
  public void prediction(int treeId, int instanceId, int prediction) {
    if (prediction != -1) {
      predictions[instanceId][prediction]++;
    }
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ForestPredictions)) {
      return false;
    }
    
    ForestPredictions fp = (ForestPredictions) obj;
    
    if (predictions.length != fp.predictions.length) {
      return false;
    }
    
    for (int i = 0; i < predictions.length; i++) {
      if (!Arrays.equals(predictions[i], fp.predictions[i])) {
        return false;
      }
    }
    
    return true;
  }
  
  @Override
  public int hashCode() {
    int hashCode = 1;
    for (int[] row : predictions) {
      for (int value : row) {
        hashCode = 31 * hashCode + value;
      }
    }
    return hashCode;
  }
  
  /**
   * compute the prediction for each instance. the prediction of an instance is the index of the label that
   * got most of the votes
   */
  public int[] computePredictions(Random rng) {
    int[] result = new int[predictions.length];
    Arrays.fill(result, -1);
    
    for (int index = 0; index < predictions.length; index++) {
      if (DataUtils.sum(predictions[index]) == 0) {
        continue; // this instance has not been classified
      }
      
      result[index] = DataUtils.maxindex(rng, predictions[index]);
    }
    
    return result;
  }
  
}
