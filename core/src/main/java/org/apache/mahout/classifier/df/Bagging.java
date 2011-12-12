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

package org.apache.mahout.classifier.df;

import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.node.Node;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;

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
   */
  public Node build(Random rng) {
    log.debug("Bagging...");
    Arrays.fill(sampled, false);
    Data bag = data.bagging(rng, sampled);
    
    log.debug("Building...");
    return treeBuilder.build(rng, bag);
  }
  
}
