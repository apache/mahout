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

package org.apache.mahout.df.ref;

import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.mahout.df.Bagging;
import org.apache.mahout.df.DecisionForest;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds a Random Decision Forest using a given TreeBuilder to grow the trees
 */
public class SequentialBuilder {
  
  private static final Logger log = LoggerFactory.getLogger(SequentialBuilder.class);
  
  private final Random rng;
  
  private final Bagging bagging;
  
  /**
   * Constructor
   * 
   * @param rng
   *          random-numbers generator
   * @param treeBuilder
   *          tree builder
   * @param data
   *          training data
   */
  public SequentialBuilder(Random rng, TreeBuilder treeBuilder, Data data) {
    this.rng = rng;
    bagging = new Bagging(treeBuilder, data);
  }
  
  public DecisionForest build(int nbTrees, PredictionCallback callback) {
    List<Node> trees = Lists.newArrayList();
    
    for (int treeId = 0; treeId < nbTrees; treeId++) {
      trees.add(bagging.build(treeId, rng, callback));
      logProgress(((float) treeId + 1) / nbTrees);
    }
    
    return new DecisionForest(trees);
  }
  
  private static void logProgress(float progress) {
    int percent = (int) (progress * 100);
    if (percent % 10 == 0) {
      log.info("Building {}%", percent);
    }
    
  }
  
}
