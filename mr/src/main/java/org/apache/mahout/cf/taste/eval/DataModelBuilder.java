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

package org.apache.mahout.cf.taste.eval;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * <p>
 * Implementations of this inner interface are simple helper classes which create a {@link DataModel} to be
 * used while evaluating a {@link org.apache.mahout.cf.taste.recommender.Recommender}.
 * 
 * @see RecommenderBuilder
 * @see RecommenderEvaluator
 */
public interface DataModelBuilder {
  
  /**
   * <p>
   * Builds a {@link DataModel} implementation to be used in an evaluation, given training data.
   * </p>
   * 
   * @param trainingData
   *          data to be used in the {@link DataModel}
   * @return {@link DataModel} based upon the given data
   */
  DataModel buildDataModel(FastByIDMap<PreferenceArray> trainingData);
  
}
