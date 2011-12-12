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

package org.apache.mahout.classifier.df.builder;

import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.node.Node;

import java.util.Random;

/**
 * Abstract base class for TreeBuilders
 */
public interface TreeBuilder {
  
  /**
   * Builds a Decision tree using the training data
   * 
   * @param rng
   *          random-numbers generator
   * @param data
   *          training data
   * @return root Node
   */
  Node build(Random rng, Data data);
  
}
