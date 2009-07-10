package org.apache.mahout.clustering.dirichlet.models;

import org.apache.hadoop.io.Writable;

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

/**
 * A model is a probability distribution over observed data points and allows the probability of any data point to be
 * computed.
 */
public interface Model<O> extends Writable {

  /**
   * Observe the given observation, retaining information about it
   *
   * @param x an Observation from the posterior
   */
  void observe(O x);

  /** Compute a new set of posterior parameters based upon the Observations that have been observed since my creation */
  void computeParameters();

  /**
   * Return the probability that the observation is described by this model
   *
   * @param x an Observation from the posterior
   * @return the probability that x is in the receiver
   */
  double pdf(O x);

  /**
   * Return the number of observations that have been observed by this model
   *
   * @return an int
   */
  int count();
}
