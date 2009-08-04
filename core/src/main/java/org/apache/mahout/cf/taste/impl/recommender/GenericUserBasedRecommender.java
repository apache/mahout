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
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.Pair;
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
import java.util.Set;

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
  public List<RecommendedItem> recommend(Comparable<?> userID, int howMany, Rescorer<Comparable<?>> rescorer)
      throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    Collection<Comparable<?>> theNeighborhood = neighborhood.getUserNeighborhood(userID);
    log.trace("UserNeighborhood is: {}", neighborhood);

    if (theNeighborhood.isEmpty()) {
      return Collections.emptyList();
    }

    Set<Comparable<?>> allItemIDs = getAllOtherItems(theNeighborhood, userID);
    log.trace("Items in neighborhood which user doesn't prefer already are: {}", allItemIDs);

    TopItems.Estimator<Comparable<?>> estimator = new Estimator(userID, theNeighborhood);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs, rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  @Override
  public float estimatePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
    DataModel model = getDataModel();
    Float actualPref = model.getPreferenceValue(userID, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    Collection<Comparable<?>> theNeighborhood = neighborhood.getUserNeighborhood(userID);
    return doEstimatePreference(userID, theNeighborhood, itemID);
  }

  @Override
  public List<Comparable<?>> mostSimilarUserIDs(Comparable<?> userID, int howMany) throws TasteException {
    return mostSimilarUserIDs(userID, howMany, null);
  }

  @Override
  public List<Comparable<?>> mostSimilarUserIDs(Comparable<?> userID,
                                                int howMany,
                                                Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer)
          throws TasteException {
    TopItems.Estimator<Comparable<?>> estimator = new MostSimilarEstimator(userID, similarity, rescorer);
    return doMostSimilarUsers(howMany, estimator);
  }

  private List<Comparable<?>> doMostSimilarUsers(int howMany,
                                                 TopItems.Estimator<Comparable<?>> estimator) throws TasteException {
    DataModel model = getDataModel();
    return TopItems.getTopUsers(howMany, model.getUserIDs(), null, estimator);
  }

  protected float doEstimatePreference(Comparable<?> theUserID,
                                       Collection<Comparable<?>> theNeighborhood,
                                       Comparable<?> itemID)
      throws TasteException {
    if (theNeighborhood.isEmpty()) {
      return Float.NaN;
    }
    DataModel dataModel = getDataModel();
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (Comparable<?> userID : theNeighborhood) {
      if (!userID.equals(theUserID)) {
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

  protected Set<Comparable<?>> getAllOtherItems(Iterable<Comparable<?>> theNeighborhood, Comparable<?> theUserID)
          throws TasteException {
    DataModel dataModel = getDataModel();
    Set<Comparable<?>> allItemIDs = new FastSet<Comparable<?>>();
    for (Comparable<?> userID : theNeighborhood) {
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      int size = prefs.length();
      for (int i = 0; i < size; i++) {
        Comparable<?> itemID = prefs.getItemID(i);
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

  private static class MostSimilarEstimator implements TopItems.Estimator<Comparable<?>> {

    private final Comparable<?> toUserID;
    private final UserSimilarity similarity;
    private final Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer;

    private MostSimilarEstimator(Comparable<?> toUserID,
                                 UserSimilarity similarity,
                                 Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer) {
      this.toUserID = toUserID;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Comparable<?> userID) throws TasteException {
      // Don't consider the user itself as a possible most similar user
      if (userID.equals(toUserID)) {
        return Double.NaN;
      }
      if (rescorer == null) {
        return similarity.userSimilarity(toUserID, userID);
      } else {
        Pair<Comparable<?>, Comparable<?>> pair = new Pair<Comparable<?>, Comparable<?>>(toUserID, userID);
        if (rescorer.isFiltered(pair)) {
          return Double.NaN;
        }
        double originalEstimate = similarity.userSimilarity(toUserID, userID);
        return rescorer.rescore(pair, originalEstimate);
      }
    }
  }

  private final class Estimator implements TopItems.Estimator<Comparable<?>> {

    private final Comparable<?> theUserUD;
    private final Collection<Comparable<?>> theNeighborhood;

    Estimator(Comparable<?> theUserUD, Collection<Comparable<?>> theNeighborhood) {
      this.theUserUD = theUserUD;
      this.theNeighborhood = theNeighborhood;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      return doEstimatePreference(theUserUD, theNeighborhood, itemID);
    }
  }
}
