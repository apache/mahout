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
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * <p>A simple {@link org.apache.mahout.cf.taste.recommender.Recommender} which uses a given
 * {@link org.apache.mahout.cf.taste.model.DataModel} and {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity}
 * to produce recommendations. This class represents Taste's support for item-based recommenders.</p>
 *
 * <p>The {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity} is the most important point to discuss here.
 * Item-based recommenders are useful because they can take advantage of something to be very fast: they base
 * their computations on item similarity, not user similarity, and item similarity is relatively static. It can be
 * precomputed, instead of re-computed in real time.</p>
 *
 * <p>Thus it's strongly recommended that you use
 * {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity}
 * with pre-computed similarities if you're going to use this class. You can use
 * {@link org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity} too, 
 * which computes similarities in real-time,
 * but will probably find this painfully slow for large amounts of data.</p>
 */
public final class GenericItemBasedRecommender extends AbstractRecommender implements ItemBasedRecommender {

  private static final Logger log = LoggerFactory.getLogger(GenericItemBasedRecommender.class);

  private final ItemSimilarity similarity;
  private final RefreshHelper refreshHelper;

  public GenericItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity) {
    super(dataModel);
    if (similarity == null) {
      throw new IllegalArgumentException("similarity is null");
    }
    this.similarity = similarity;
    this.refreshHelper = new RefreshHelper(null);
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(similarity);
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
    if (getNumPreferences(theUser) == 0) {
      return Collections.emptyList();
    }

    Set<Item> allItems = getAllOtherItems(theUser);

    TopItems.Estimator<Item> estimator = new Estimator(theUser);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItems, rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  @Override
  public double estimatePreference(Object userID, Object itemID) throws TasteException {
    DataModel model = getDataModel();
    User theUser = model.getUser(userID);
    Preference actualPref = theUser.getPreferenceFor(itemID);
    if (actualPref != null) {
      return actualPref.getValue();
    }
    Item item = model.getItem(itemID);
    return doEstimatePreference(theUser, item);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(Object itemID, int howMany) throws TasteException {
    return mostSimilarItems(itemID, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(Object itemID,
                                                int howMany,
                                                Rescorer<Pair<Item, Item>> rescorer) throws TasteException {
    Item toItem = getDataModel().getItem(itemID);
    TopItems.Estimator<Item> estimator = new MostSimilarEstimator(toItem, similarity, rescorer);
    return doMostSimilarItems(itemID, howMany, estimator);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(List<Object> itemIDs, int howMany) throws TasteException {
    return mostSimilarItems(itemIDs, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(List<Object> itemIDs,
                                                int howMany,
                                                Rescorer<Pair<Item, Item>> rescorer) throws TasteException {
    DataModel model = getDataModel();
    List<Item> toItems = new ArrayList<Item>(itemIDs.size());
    for (Object itemID : itemIDs) {
      toItems.add(model.getItem(itemID));
    }
    TopItems.Estimator<Item> estimator = new MultiMostSimilarEstimator(toItems, similarity, rescorer);
    Collection<Item> allItems = new FastSet<Item>(model.getNumItems());
    for (Item item : model.getItems()) {
      allItems.add(item);
    }
    for (Item item : toItems) {
      allItems.remove(item);
    }
    return TopItems.getTopItems(howMany, allItems, null, estimator);
  }

  @Override
  public List<RecommendedItem> recommendedBecause(Object userID,
                                                  Object itemID,
                                                  int howMany) throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    if (itemID == null) {
      throw new IllegalArgumentException("itemID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    DataModel model = getDataModel();
    User user = model.getUser(userID);
    Item recommendedItem = model.getItem(itemID);
    TopItems.Estimator<Item> estimator = new RecommendedBecauseEstimator(user, recommendedItem, similarity);

    Collection<Item> allUserItems = new FastSet<Item>();
    Preference[] prefs = user.getPreferencesAsArray();
    for (int i = 0; i < prefs.length; i++) {
      allUserItems.add(prefs[i].getItem());
    }
    allUserItems.remove(recommendedItem);

    return TopItems.getTopItems(howMany, allUserItems, null, estimator);
  }

  private List<RecommendedItem> doMostSimilarItems(Object itemID,
                                                   int howMany,
                                                   TopItems.Estimator<Item> estimator) throws TasteException {
    DataModel model = getDataModel();
    Item toItem = model.getItem(itemID);
    Collection<Item> allItems = new FastSet<Item>(model.getNumItems());
    for (Item item : model.getItems()) {
      allItems.add(item);
    }
    allItems.remove(toItem);
    return TopItems.getTopItems(howMany, allItems, null, estimator);
  }

  private double doEstimatePreference(User theUser, Item item) throws TasteException {
    double preference = 0.0;
    double totalSimilarity = 0.0;
    Preference[] prefs = theUser.getPreferencesAsArray();
    for (int i = 0; i < prefs.length; i++) {
      Preference pref = prefs[i];
      double theSimilarity = similarity.itemSimilarity(item, pref.getItem());
      if (!Double.isNaN(theSimilarity)) {
        // Why + 1.0? similarity ranges from -1.0 to 1.0, and we want to use it as a simple
        // weight. To avoid negative values, we add 1.0 to put it in
        // the [0.0,2.0] range which is reasonable for weights
        theSimilarity += 1.0;
        preference += theSimilarity * pref.getValue();
        totalSimilarity += theSimilarity;
      }
    }
    return totalSimilarity == 0.0 ? Double.NaN : preference / totalSimilarity;
  }

  private static int getNumPreferences(User theUser) {
    return theUser.getPreferencesAsArray().length;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "GenericItemBasedRecommender[similarity:" + similarity + ']';
  }

  private static class MostSimilarEstimator implements TopItems.Estimator<Item> {

    private final Item toItem;
    private final ItemSimilarity similarity;
    private final Rescorer<Pair<Item, Item>> rescorer;

    private MostSimilarEstimator(Item toItem,
                                 ItemSimilarity similarity,
                                 Rescorer<Pair<Item, Item>> rescorer) {
      this.toItem = toItem;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Item item) throws TasteException {
      Pair<Item, Item> pair = new Pair<Item, Item>(toItem, item);
      if (rescorer != null && rescorer.isFiltered(pair)) {
        return Double.NaN;
      }
      double originalEstimate = similarity.itemSimilarity(toItem, item);
      return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
    }
  }

  private final class Estimator implements TopItems.Estimator<Item> {

    private final User theUser;

    private Estimator(User theUser) {
      this.theUser = theUser;
    }

    @Override
    public double estimate(Item item) throws TasteException {
      return doEstimatePreference(theUser, item);
    }
  }

  private static class MultiMostSimilarEstimator implements TopItems.Estimator<Item> {

    private final List<Item> toItems;
    private final ItemSimilarity similarity;
    private final Rescorer<Pair<Item, Item>> rescorer;

    private MultiMostSimilarEstimator(List<Item> toItems,
                                      ItemSimilarity similarity,
                                      Rescorer<Pair<Item, Item>> rescorer) {
      this.toItems = toItems;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Item item) throws TasteException {
      RunningAverage average = new FullRunningAverage();
      for (Item toItem : toItems) {
        Pair<Item, Item> pair = new Pair<Item, Item>(toItem, item);
        if (rescorer != null && rescorer.isFiltered(pair)) {
          continue;
        }
        double estimate = similarity.itemSimilarity(toItem, item);
        if (rescorer != null) {
          estimate = rescorer.rescore(pair, estimate);
        }
        average.addDatum(estimate);
      }
      return average.getAverage();
    }
  }

  private static class RecommendedBecauseEstimator implements TopItems.Estimator<Item> {

    private final User user;
    private final Item recommendedItem;
    private final ItemSimilarity similarity;

    private RecommendedBecauseEstimator(User user,
                                        Item recommendedItem,
                                        ItemSimilarity similarity) {
      this.user = user;
      this.recommendedItem = recommendedItem;
      this.similarity = similarity;
    }

    @Override
    public double estimate(Item item) throws TasteException {
      Preference pref = user.getPreferenceFor(item.getID());
      if (pref == null) {
        return Double.NaN;
      }
      double similarityValue = similarity.itemSimilarity(recommendedItem, item);
      return (1.0 + similarityValue) * pref.getValue();
    }
  }

}
