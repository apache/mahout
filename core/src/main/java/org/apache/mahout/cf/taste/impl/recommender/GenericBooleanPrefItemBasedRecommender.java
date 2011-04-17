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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.MostSimilarItemsCandidateItemsStrategy;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

/**
 * A variant on {@link GenericItemBasedRecommender} which is appropriate for use when no notion of preference
 * value exists in the data.
 *
 * @see org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender
 */
public final class GenericBooleanPrefItemBasedRecommender extends GenericItemBasedRecommender {

  public GenericBooleanPrefItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity) {
    super(dataModel, similarity);
  }

  public GenericBooleanPrefItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity,
      CandidateItemsStrategy candidateItemsStrategy, MostSimilarItemsCandidateItemsStrategy
      mostSimilarItemsCandidateItemsStrategy) {
    super(dataModel, similarity, candidateItemsStrategy, mostSimilarItemsCandidateItemsStrategy);
  }
  
  /**
   * This computation is in a technical sense, wrong, since in the domain of "boolean preference users" where
   * all preference values are 1, this method should only ever return 1.0 or NaN. This isn't terribly useful
   * however since it means results can't be ranked by preference value (all are 1). So instead this returns a
   * sum of similarities.
   */
  @Override
  protected float doEstimatePreference(long userID, PreferenceArray preferencesFromUser, long itemID)
    throws TasteException {
    double[] similarities = getSimilarity().itemSimilarities(itemID, preferencesFromUser.getIDs());
    boolean foundAPref = false;
    double totalSimilarity = 0.0;
    for (double theSimilarity : similarities) {
      if (!Double.isNaN(theSimilarity)) {
        foundAPref = true;
        totalSimilarity += theSimilarity;
      }
    }
    return foundAPref ? (float) totalSimilarity : Float.NaN;
  }
  
  @Override
  public String toString() {
    return "GenericBooleanPrefItemBasedRecommender";
  }
  
}
