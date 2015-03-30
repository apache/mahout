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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * <p>
 * Implementations of this inner interface are simple helper classes which create a {@link Recommender} to be
 * evaluated based on the given {@link DataModel}.
 * </p>
 */
public interface RecommenderBuilder {
  
  /**
   * <p>
   * Builds a {@link Recommender} implementation to be evaluated, using the given {@link DataModel}.
   * </p>
   * 
   * @param dataModel
   *          {@link DataModel} to build the {@link Recommender} on
   * @return {@link Recommender} based upon the given {@link DataModel}
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  Recommender buildRecommender(DataModel dataModel) throws TasteException;
  
}
