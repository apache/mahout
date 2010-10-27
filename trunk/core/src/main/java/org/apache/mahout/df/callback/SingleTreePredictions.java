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

import com.google.common.base.Preconditions;

/**
 * Collects the predictions for a single tree
 */
public class SingleTreePredictions implements PredictionCallback {
  
  /** predictions[n] = 'label' predicted for instance 'n' */
  private final int[] predictions;
  
  /** used to assert that all the predictions belong to the same tree */
  private Integer treeId;
  
  public SingleTreePredictions(int nbInstances) {
    predictions = new int[nbInstances];
    Arrays.fill(predictions, -1);
  }
  
  @Override
  public void prediction(int treeId, int instanceId, int prediction) {
    if (this.treeId == null) {
      this.treeId = treeId;
    } else {
      Preconditions.checkArgument(this.treeId == treeId, "the predictions does not belong to the same tree");
    }
    
    predictions[instanceId] = prediction;
  }
  
  public int[] getPredictions() {
    return predictions;
  }
  
}
