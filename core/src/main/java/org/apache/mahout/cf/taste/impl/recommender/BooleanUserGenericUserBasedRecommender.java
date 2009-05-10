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
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.model.BooleanPrefUser;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.Queue;
import java.util.PriorityQueue;
import java.util.ArrayList;

/**
 * A variant on {@link GenericUserBasedRecommender} which is appropriate
 * for use with "boolean" classes like {@link org.apache.mahout.cf.taste.impl.model.BooleanPrefUser}.
 */
public final class BooleanUserGenericUserBasedRecommender extends AbstractRecommender implements UserBasedRecommender {

  private static final Logger log = LoggerFactory.getLogger(BooleanUserGenericUserBasedRecommender.class);

  private final UserNeighborhood neighborhood;
  private final UserSimilarity similarity;
  private final RefreshHelper refreshHelper;

  public BooleanUserGenericUserBasedRecommender(DataModel dataModel,
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
  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
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

    Set<Object> allItems = getAllOtherItems(theNeighborhood, theUser);
    log.trace("Items in neighborhood which user doesn't prefer already are: {}", allItems);

    TopItems.Estimator<Object> estimator = new Estimator(theUser, theNeighborhood);

    List<RecommendedItem> topItems = getTopItems(howMany, allItems, rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  public static List<RecommendedItem> getTopItems(int howMany,
                                                  Iterable<Object> allItems,
                                                  Rescorer<Item> rescorer,
                                                  TopItems.Estimator<Object> estimator) throws TasteException {
    if (allItems == null || estimator == null) {
      throw new IllegalArgumentException("argument is null");
    }
    Queue<RecommendedItem> topItems = new PriorityQueue<RecommendedItem>(howMany + 1, Collections.reverseOrder());
    boolean full = false;
    double lowestTopValue = Double.NEGATIVE_INFINITY;
    for (Object itemID : allItems) {
        double preference = estimator.estimate(itemID);
        double rescoredPref = rescorer == null ? preference : rescorer.rescore(new GenericItem<String>(itemID.toString()), preference);
        if (!Double.isNaN(rescoredPref) && (!full || rescoredPref > lowestTopValue)) {
          topItems.add(new GenericRecommendedItem(new GenericItem<String>(itemID.toString()), rescoredPref));
          if (full) {
            topItems.poll();
          } else if (topItems.size() > howMany) {
            full = true;
            topItems.poll();
          }
          lowestTopValue = topItems.peek().getValue();
        }
    }
    List<RecommendedItem> result = new ArrayList<RecommendedItem>(topItems.size());
    result.addAll(topItems);
    Collections.sort(result);
    return result;
  }

  @Override
  public double estimatePreference(Object userID, Object itemID) throws TasteException {
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
  public List<User> mostSimilarUsers(Object userID, int howMany) throws TasteException {
    return mostSimilarUsers(userID, howMany, null);
  }

  @Override
  public List<User> mostSimilarUsers(Object userID,
                                     int howMany,
                                     Rescorer<Pair<User, User>> rescorer) throws TasteException {
    User toUser = getDataModel().getUser(userID);
    TopItems.Estimator<User> estimator = new MostSimilarEstimator(toUser, similarity, rescorer);
    return doMostSimilarUsers(userID, howMany, estimator);
  }

  private List<User> doMostSimilarUsers(Object userID,
                                        int howMany,
                                        TopItems.Estimator<User> estimator) throws TasteException {
    DataModel model = getDataModel();
    User toUser = model.getUser(userID);
    Collection<User> allUsers = new FastSet<User>(model.getNumUsers());
    for (User user : model.getUsers()) {
      allUsers.add(user);
    }
    allUsers.remove(toUser);
    return TopItems.getTopUsers(howMany, allUsers, null, estimator);
  }

  /**
   * This computation is in a technical sense, wrong, since in the domain of "boolean preference users"
   * where all preference values are 1, this method should only ever return 1.0 or NaN. This isn't
   * terribly useful however since it means results can't be ranked by preference value (all are 1).
   * So instead this returns a sum of similarties to any other user in the neighborhood who has also
   * rated the item.
   */
  private double doEstimatePreference(User theUser, Collection<User> theNeighborhood, Object itemID)
          throws TasteException {
    if (theNeighborhood.isEmpty()) {
      return Double.NaN;
    }
    double totalSimilarity = 0.0;
    boolean foundAPref = false;
    for (User user : theNeighborhood) {
      if (!user.equals(theUser)) {
        // See GenericItemBasedRecommender.doEstimatePreference() too
        if (user.getPreferenceFor(itemID) != null) {
          foundAPref = true;
          totalSimilarity += similarity.userSimilarity(theUser, user);
        }
      }
    }
    return foundAPref ? totalSimilarity : 0.0;
  }

  private static Set<Object> getAllOtherItems(Iterable<User> theNeighborhood, User theUser) {
    Set<Object> allItems = new FastSet<Object>();
    for (User user : theNeighborhood) {
      allItems.addAll(((BooleanPrefUser<?>) user).getItemIDs());
    }
    allItems.removeAll(((BooleanPrefUser<?>) theUser).getItemIDs());
    return allItems;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "BooleanUserGenericUserBasedRecommender[neighborhood:" + neighborhood + ']';
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
      Pair<User, User> pair = new Pair<User, User>(toUser, user);
      if (rescorer != null && rescorer.isFiltered(pair)) {
        return Double.NaN;
      }
      double originalEstimate = similarity.userSimilarity(toUser, user);
      return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
    }
  }

  private final class Estimator implements TopItems.Estimator<Object> {

    private final User theUser;
    private final Collection<User> theNeighborhood;

    Estimator(User theUser, Collection<User> theNeighborhood) {
      this.theUser = theUser;
      this.theNeighborhood = theNeighborhood;
    }

    @Override
    public double estimate(Object itemID) throws TasteException {
      return doEstimatePreference(theUser, theNeighborhood, itemID);
    }
  }
}
