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

package org.apache.mahout.cf.taste.impl.recommender.slim;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.math.Matrix;

/**
 * SLIM Optimizer interface. The optimzation algorithm will learn W,
 * the item-to-item sparse coefficient matrix used to produce
 * recommendations.
 *
 */
public interface Optimizer extends Refreshable {

  /**
   * Implementation must be able to create a SlimSolution.
   */
  SlimSolution findSolution() throws TasteException;

  /**
   * Used to get values from the matrix the optimizer is learning.
   * The optimizer should also intialize the value (to preserve sparsity)
   */
  double getAndInitWeight(Matrix itemWeights, int row, int column);
  
  /**
   * Used to get values from the matrix the optimizer is learning.
   * The optimizer should also intialize with positive value.
   */
  double getAndInitWeightPos(Matrix itemWeights, int row, int column);
  
}
