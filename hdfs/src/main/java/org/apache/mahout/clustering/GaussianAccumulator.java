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

public interface GaussianAccumulator {

  /**
   * @return the number of observations
   */
  double getN();

  /**
   * @return the mean of the observations
   */
  Vector getMean();

  /**
   * @return the std of the observations
   */
  Vector getStd();
  
  /**
   * @return the average of the vector std elements
   */
  double getAverageStd();
  
  /**
   * @return the variance of the observations
   */
  Vector getVariance();

  /**
   * Observe the vector 
   * 
   * @param x a Vector
   * @param weight the double observation weight (usually 1.0)
   */
  void observe(Vector x, double weight);

  /**
   * Compute the mean, variance and standard deviation
   */
  void compute();

}
