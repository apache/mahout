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

package org.apache.mahout.df;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.DataUtils;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.node.Node;

/**
 * Represents a forest of decision trees.
 */
public class DecisionForest {

  private final List<Node> trees;

  protected DecisionForest() {
    trees = new ArrayList<Node>();
  }

  public DecisionForest(List<Node> trees) {
    if (!(trees != null && !trees.isEmpty())) {
      throw new IllegalArgumentException("trees argument must not be null or empty");
    }

    this.trees = trees;
  }

  public List<Node> getTrees() {
    return trees;
  }
  
  /**
   * Classifies the data and calls callback for each classification
   * 
   * @param data
   * @param callback
   */
  public void classify(Data data, PredictionCallback callback) {
    if (callback == null) {
      throw new IllegalArgumentException("callback must not be null");
    }

    if (data.isEmpty())
      return; // nothing to classify

    for (int treeId = 0; treeId < trees.size(); treeId++) {
      Node tree = trees.get(treeId);

      for (int index = 0; index < data.size(); index++) {
        int prediction = tree.classify(data.get(index));
        callback.prediction(treeId, index, prediction);
      }
    }
  }

  /**
   * predicts the label for the instance
   * 
   * @param rng Random number generator, used to break ties randomly
   * @param instance
   * @return -1 if the label cannot be predicted
   */
  public int classify(Random rng, Instance instance) {
    int[] predictions = new int[trees.size()];

    for (Node tree : trees) {
      int prediction = tree.classify(instance);
      if (prediction != -1)
        predictions[prediction]++;
    }

    if (DataUtils.sum(predictions) == 0)
      return -1; // no prediction available

    return DataUtils.maxindex(rng, predictions);
  }

  /**
   * Mean number of nodes per tree
   * 
   * @return
   */
  public long meanNbNodes() {
    long sum = 0;

    for (Node tree : trees) {
      sum += tree.nbNodes();
    }

    return sum / trees.size();
  }

  /**
   * Total number of nodes in all the trees
   * 
   * @return
   */
  public long nbNodes() {
    long sum = 0;

    for (Node tree : trees) {
      sum += tree.nbNodes();
    }

    return sum;
  }

  /**
   * Mean maximum depth per tree
   * 
   * @return
   */
  public long meanMaxDepth() {
    long sum = 0;

    for (Node tree : trees) {
      sum += tree.maxDepth();
    }

    return sum / trees.size();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj==null || !(obj instanceof DecisionForest))
      return false;
    
    DecisionForest rf = (DecisionForest)obj;
    
    return trees.size() == rf.getTrees().size() && trees.containsAll(rf.getTrees());
  }

  @Override
  public int hashCode() {
    return trees.hashCode();
  }

}
