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
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
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
public final class GenericUserBasedRecommender extends AbstractRecommender implements UserBasedRecommender {

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

    User theUser = getDataModel().getUser(userID);
    Collection<User> theNeighborhood = neighborhood.getUserNeighborhood(userID);
    log.trace("UserNeighborhood is: {}", neighborhood);

    if (theNeighborhood.isEmpty()) {
      return Collections.emptyList();
    }

    Set<Comparable<?>> allItemIDs = getAllOtherItems(theNeighborhood, theUser);
    log.trace("Items in neighborhood which user doesn't prefer already are: {}", allItemIDs);

    TopItems.Estimator<Comparable<?>> estimator = new Estimator(theUser, theNeighborhood);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs, rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  @Override
  public double estimatePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
    DataModel model = getDataModel();
    User theUser = model.getUser(userID);
    Preference actualPref = theUser.getPreferenceFor(itemID);
    if (actualPref != null) {
      return actualPref.getValue();
    }
    Collection<User> theNeighborhood = neighborhood.getUserNeighborhood(userID);
    return doEstimatePreference(theUser, theNeighborhood, itemID);
  }

  @Override
  public List<User> mostSimilarUsers(Comparable<?> userID, int howMany) throws TasteException {
    return mostSimilarUsers(userID, howMany, null);
  }

  @Override
  public List<User> mostSimilarUsers(Comparable<?> userID,
                                     int howMany,
                                     Rescorer<Pair<User, User>> rescorer) throws TasteException {
    User toUser = getDataModel().getUser(userID);
    TopItems.Estimator<User> estimator = new MostSimilarEstimator(toUser, similarity, rescorer);
    return doMostSimilarUsers(howMany, estimator);
  }

  private List<User> doMostSimilarUsers(int howMany,
                                        TopItems.Estimator<User> estimator) throws TasteException {
    DataModel model = getDataModel();
    return TopItems.getTopUsers(howMany, model.getUsers(), null, estimator);
  }

  private double doEstimatePreference(User theUser, Collection<User> theNeighborhood, Comparable<?> itemID)
      throws TasteException {
    if (theNeighborhood.isEmpty()) {
      return Double.NaN;
    }
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (User user : theNeighborhood) {
      if (!user.equals(theUser)) {
        // See GenericItemBasedRecommender.doEstimatePreference() too
        Preference pref = user.getPreferenceFor(itemID);
        if (pref != null) {
          double theSimilarity = similarity.userSimilarity(theUser, user) + 1.0;
          // Similarity should not be NaN or else the user should never have showed up
          // in the neighborhood. Adding 1.0 puts this in the range [0,2] which is
          // more appropriate for weights
          preference += theSimilarity * pref.getValue();
          totalSimilarity += theSimilarity;
        }
      }
    }
    return totalSimilarity == 0.0 ? Double.NaN : preference / totalSimilarity;
  }

  private static Set<Comparable<?>> getAllOtherItems(Iterable<User> theNeighborhood, User theUser) {
    Set<Comparable<?>> allItemIDs = new FastSet<Comparable<?>>();
    for (User user : theNeighborhood) {
      Preference[] prefs = user.getPreferencesAsArray();
      for (Preference pref : prefs) {
        Comparable<?> itemID = pref.getItemID();
        // If not already preferred by the user, add it
        if (theUser.getPreferenceFor(itemID) == null) {
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

  private static class MostSimilarEstimator implements TopItems.Estimator<User> {

    private final User toUser;
    private final UserSimilarity similarity;
    private final Rescorer<Pair<User, User>> rescorer;

    private MostSimilarEstimator(User toUser,
                                 UserSimilarity similarity,
                                 Rescorer<Pair<User, User>> rescorer) {
      this.toUser = toUser;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(User user) throws TasteException {
      // Don't consider the user itself as a possible most similar user
      if (user.equals(toUser)) {
        return Double.NaN;
      }
      Pair<User, User> pair = new Pair<User, User>(toUser, user);
      if (rescorer != null && rescorer.isFiltered(pair)) {
        return Double.NaN;
      }
      double originalEstimate = similarity.userSimilarity(toUser, user);
      return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
    }
  }

  private final class Estimator implements TopItems.Estimator<Comparable<?>> {

    private final User theUser;
    private final Collection<User> theNeighborhood;

    Estimator(User theUser, Collection<User> theNeighborhood) {
      this.theUser = theUser;
      this.theNeighborhood = theNeighborhood;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      return doEstimatePreference(theUser, theNeighborhood, itemID);
    }
  }
}
