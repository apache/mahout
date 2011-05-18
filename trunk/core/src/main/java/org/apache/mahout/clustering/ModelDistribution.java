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

/** A model distribution allows us to sample a model from its prior distribution. */
public interface ModelDistribution<O> {
  
  /**
   * Return a list of models sampled from the prior
   * 
   * @param howMany
   *          the int number of models to return
   * @return a Model<Observation>[] representing what is known apriori
   */
  Model<O>[] sampleFromPrior(int howMany);
  
  /**
   * Return a list of models sampled from the posterior
   * 
   * @param posterior
   *          the Model<Observation>[] after observations
   * @return a Model<Observation>[] representing what is known apriori
   */
  Model<O>[] sampleFromPosterior(Model<O>[] posterior);
  
}
