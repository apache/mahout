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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * <p>A simple {@link Recommender} which uses a given {@link DataModel} and {@link UserNeighborhood} to produce
 * recommendations.</p>
 */
public class GenericUserBasedRecommender extends AbstractRecommender implements UserBasedRecommender {

  private static final Logger log = LoggerFactory.getLogger(GenericUserBasedRecommender.class);

  private final UserNeighborhood neighborhood;
  private final UserSimilarity similarity;
  private final RefreshHelper refreshHelper;

  public GenericUserBasedRecommender(DataModel dataModel,
                                     UserNeighborhood neighborhood,
                                     UserSimilarity similarity) {
    super(dataModel);
    if (neighborhood == null) {
      throw new IllegalArgumentException("neighborhood is null");
    }
    this.neighborhood = neighborhood;
    this.similarity = similarity;
    this.refreshHelper = new RefreshHelper(null);
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(similarity);
    refreshHelper.addDependency(neighborhood);
  }

  public UserSimilarity getSimilarity() {
    return similarity;
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer)
      throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    long[] theNeighborhood = neighborhood.getUserNeighborhood(userID);

    if (theNeighborhood.length == 0) {
      return Collections.emptyList();
    }

    FastIDSet allItemIDs = getAllOtherItems(theNeighborhood, userID);

    TopItems.Estimator<Long> estimator = new Estimator(userID, theNeighborhood);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs.iterator(), rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    DataModel model = getDataModel();
    Float actualPref = model.getPreferenceValue(userID, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    long[] theNeighborhood = neighborhood.getUserNeighborhood(userID);
    return doEstimatePreference(userID, theNeighborhood, itemID);
  }

  @Override
  public long[] mostSimilarUserIDs(long userID, int howMany) throws TasteException {
    return mostSimilarUserIDs(userID, howMany, null);
  }

  @Override
  public long[] mostSimilarUserIDs(long userID, int howMany, Rescorer<LongPair> rescorer)
          throws TasteException {
    TopItems.Estimator<Long> estimator = new MostSimilarEstimator(userID, similarity, rescorer);
    return doMostSimilarUsers(howMany, estimator);
  }

  private long[] doMostSimilarUsers(int howMany, TopItems.Estimator<Long> estimator) throws TasteException {
    DataModel model = getDataModel();
    return TopItems.getTopUsers(howMany, model.getUserIDs(), null, estimator);
  }

  protected float doEstimatePreference(long theUserID, long[] theNeighborhood, long itemID)
      throws TasteException {
    if (theNeighborhood.length == 0) {
      return Float.NaN;
    }
    DataModel dataModel = getDataModel();
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (long userID : theNeighborhood) {
      if (userID != theUserID) {
        // See GenericItemBasedRecommender.doEstimatePreference() too
        Float pref = dataModel.getPreferenceValue(userID, itemID);
        if (pref != null) {
          double theSimilarity = similarity.userSimilarity(theUserID, userID) + 1.0;
          // Similarity should not be NaN or else the user should never have showed up
          // in the neighborhood. Adding 1.0 puts this in the range [0,2] which is
          // more appropriate for weights
          preference += theSimilarity * pref;
          totalSimilarity += theSimilarity;
        }
      }
    }
    return totalSimilarity == 0.0 ? Float.NaN : (float) (preference / totalSimilarity);
  }

  protected FastIDSet getAllOtherItems(long[] theNeighborhood, long theUserID)
          throws TasteException {
    DataModel dataModel = getDataModel();
    FastIDSet allItemIDs = new FastIDSet();
    for (long userID : theNeighborhood) {
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      int size = prefs.length();
      for (int i = 0; i < size; i++) {
        long itemID = prefs.getItemID(i);
        // If not already preferred by the user, add it
        if (dataModel.getPreferenceValue(theUserID, itemID) == null) {
          allItemIDs.add(itemID);
        }
      }
    }
    return allItemIDs;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "GenericUserBasedRecommender[neighborhood:" + neighborhood + ']';
  }

  private static class MostSimilarEstimator implements TopItems.Estimator<Long> {

    private final long toUserID;
    private final UserSimilarity similarity;
    private final Rescorer<LongPair> rescorer;

    private MostSimilarEstimator(long toUserID,
                                 UserSimilarity similarity,
                                 Rescorer<LongPair> rescorer) {
      this.toUserID = toUserID;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Long userID) throws TasteException {
      // Don't consider the user itself as a possible most similar user
      if (userID == toUserID) {
        return Double.NaN;
      }
      if (rescorer == null) {
        return similarity.userSimilarity(toUserID, userID);
      } else {
        LongPair pair = new LongPair(toUserID, userID);
        if (rescorer.isFiltered(pair)) {
          return Double.NaN;
        }
        double originalEstimate = similarity.userSimilarity(toUserID, userID);
        return rescorer.rescore(pair, originalEstimate);
      }
    }
  }

  private final class Estimator implements TopItems.Estimator<Long> {

    private final long theUserID;
    private final long[] theNeighborhood;

    Estimator(long theUserID, long[] theNeighborhood) {
      this.theUserID = theUserID;
      this.theNeighborhood = theNeighborhood;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return doEstimatePreference(theUserID, theNeighborhood, itemID);
    }
  }
}
