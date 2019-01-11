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
import org.apache.mahout.cf.taste.recommender.IDRescorer;

/**
 * <p>
 * Implementations collect information retrieval-related statistics on a
 * {@link org.apache.mahout.cf.taste.recommender.Recommender}'s performance, including precision, recall and
 * f-measure.
 * </p>
 * 
 * <p>
 * See <a href="http://en.wikipedia.org/wiki/Information_retrieval">Information retrieval</a>.
 */
public interface RecommenderIRStatsEvaluator {
  
  /**
   * @param recommenderBuilder
   *          object that can build a {@link org.apache.mahout.cf.taste.recommender.Recommender} to test
   * @param dataModelBuilder
   *          {@link DataModelBuilder} to use, or if null, a default {@link DataModel} implementation will be
   *          used
   * @param dataModel
   *          dataset to test on
   * @param rescorer
   *          if any, to use when computing recommendations
   * @param at
   *          as in, "precision at 5". The number of recommendations to consider when evaluating precision,
   *          etc.
   * @param relevanceThreshold
   *          items whose preference value is at least this value are considered "relevant" for the purposes
   *          of computations
   * @return {@link IRStatistics} with resulting precision, recall, etc.
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  IRStatistics evaluate(RecommenderBuilder recommenderBuilder,
                        DataModelBuilder dataModelBuilder,
                        DataModel dataModel,
                        IDRescorer rescorer,
                        int at,
                        double relevanceThreshold,
                        double evaluationPercentage) throws TasteException;
  
}
