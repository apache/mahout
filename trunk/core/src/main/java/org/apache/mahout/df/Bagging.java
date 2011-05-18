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

import java.util.Arrays;
import java.util.Random;

import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds a tree using bagging
 */
public class Bagging {
  
  private static final Logger log = LoggerFactory.getLogger(Bagging.class);
  
  private final TreeBuilder treeBuilder;
  
  private final Data data;
  
  private final boolean[] sampled;
  
  public Bagging(TreeBuilder treeBuilder, Data data) {
    this.treeBuilder = treeBuilder;
    this.data = data;
    sampled = new boolean[data.size()];
  }
  
  /**
   * Builds one tree
   * 
   * @param treeId
   *          tree identifier
   */
  public Node build(int treeId, Random rng, PredictionCallback callback) {
    log.debug("Bagging...");
    Arrays.fill(sampled, false);
    Data bag = data.bagging(rng, sampled);
    
    log.debug("Building...");
    Node tree = treeBuilder.build(rng, bag);
    
    // predict the label for the out-of-bag elements
    if (callback != null) {
      log.debug("Oob error estimation");
      for (int index = 0; index < data.size(); index++) {
        if (!sampled[index]) {
          int prediction = tree.classify(data.get(index));
          callback.prediction(treeId, index, prediction);
        }
      }
    }
    
    return tree;
  }
  
}
