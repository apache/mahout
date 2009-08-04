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
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * <p>A simple {@link org.apache.mahout.cf.taste.recommender.Recommender} which uses a given {@link
 * org.apache.mahout.cf.taste.model.DataModel} and {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity} to
 * produce recommendations. This class represents Taste's support for item-based recommenders.</p>
 *
 * <p>The {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity} is the most important point to discuss here.
 * Item-based recommenders are useful because they can take advantage of something to be very fast: they base their
 * computations on item similarity, not user similarity, and item similarity is relatively static. It can be
 * precomputed, instead of re-computed in real time.</p>
 *
 * <p>Thus it's strongly recommended that you use {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity}
 * with pre-computed similarities if you're going to use this class. You can use {@link
 * org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity} too, which computes similarities in
 * real-time, but will probably find this painfully slow for large amounts of data.</p>
 */
public class GenericItemBasedRecommender extends AbstractRecommender implements ItemBasedRecommender {

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

  public ItemSimilarity getSimilarity() {
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

    if (getNumPreferences(userID) == 0) {
      return Collections.emptyList();
    }

    Set<Comparable<?>> allItemIDs = getAllOtherItems(userID);

    TopItems.Estimator<Comparable<?>> estimator = new Estimator(userID);

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
    return doEstimatePreference(userID, itemID);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(Comparable<?> itemID, int howMany) throws TasteException {
    return mostSimilarItems(itemID, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(Comparable<?> itemID,
                                                int howMany,
                                                Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer) throws TasteException {
    TopItems.Estimator<Comparable<?>> estimator = new MostSimilarEstimator(itemID, similarity, rescorer);
    return doMostSimilarItems(itemID, howMany, estimator);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(List<Comparable<?>> itemIDs, int howMany) throws TasteException {
    return mostSimilarItems(itemIDs, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(List<Comparable<?>> itemIDs,
                                                int howMany,
                                                Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer) throws TasteException {
    DataModel model = getDataModel();
    TopItems.Estimator<Comparable<?>> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, rescorer);
    Collection<Comparable<?>> allItemIDs = new FastSet<Comparable<?>>(model.getNumItems());
    for (Comparable<?> item : model.getItemIDs()) {
      allItemIDs.add(item);
    }
    allItemIDs.removeAll(itemIDs);
    return TopItems.getTopItems(howMany, allItemIDs, null, estimator);
  }

  @Override
  public List<RecommendedItem> recommendedBecause(Comparable<?> userID,
                                                  Comparable<?> itemID,
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
    TopItems.Estimator<Comparable<?>> estimator = new RecommendedBecauseEstimator(userID, itemID, similarity);

    Collection<Comparable<?>> allUserItems = new FastSet<Comparable<?>>();
    PreferenceArray prefs = model.getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      allUserItems.add(prefs.getItemID(i));
    }
    allUserItems.remove(itemID);

    return TopItems.getTopItems(howMany, allUserItems, null, estimator);
  }

  private List<RecommendedItem> doMostSimilarItems(Comparable<?> itemID,
                                                   int howMany,
                                                   TopItems.Estimator<Comparable<?>> estimator) throws TasteException {
    DataModel model = getDataModel();
    Collection<Comparable<?>> allItemIDs = new FastSet<Comparable<?>>(model.getNumItems());
    for (Comparable<?> item : model.getItemIDs()) {
      allItemIDs.add(item);
    }
    allItemIDs.remove(itemID);
    return TopItems.getTopItems(howMany, allItemIDs, null, estimator);
  }

  protected float doEstimatePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
    double preference = 0.0;
    double totalSimilarity = 0.0;
    PreferenceArray prefs = getDataModel().getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      double theSimilarity = similarity.itemSimilarity(itemID, prefs.getItemID(i));
      if (!Double.isNaN(theSimilarity)) {
        // Why + 1.0? similarity ranges from -1.0 to 1.0, and we want to use it as a simple
        // weight. To avoid negative values, we add 1.0 to put it in
        // the [0.0,2.0] range which is reasonable for weights
        theSimilarity += 1.0;
        preference += theSimilarity * prefs.getValue(i);
        totalSimilarity += theSimilarity;
      }
    }
    return totalSimilarity == 0.0 ? Float.NaN : (float) (preference / totalSimilarity);
  }

  private int getNumPreferences(Comparable<?> userID) throws TasteException {
    return getDataModel().getPreferencesFromUser(userID).length();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "GenericItemBasedRecommender[similarity:" + similarity + ']';
  }

  public static class MostSimilarEstimator implements TopItems.Estimator<Comparable<?>> {

    private final Comparable<?> toItemID;
    private final ItemSimilarity similarity;
    private final Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer;

    public MostSimilarEstimator(Comparable<?> toItemID,
                                ItemSimilarity similarity,
                                Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer) {
      this.toItemID = toItemID;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      Pair<Comparable<?>, Comparable<?>> pair = new Pair<Comparable<?>, Comparable<?>>(toItemID, itemID);
      if (rescorer != null && rescorer.isFiltered(pair)) {
        return Double.NaN;
      }
      double originalEstimate = similarity.itemSimilarity(toItemID, itemID);
      return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
    }
  }

  private final class Estimator implements TopItems.Estimator<Comparable<?>> {

    private final Comparable<?> userID;

    private Estimator(Comparable<?> userID) {
      this.userID = userID;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      return doEstimatePreference(userID, itemID);
    }
  }

  private static class MultiMostSimilarEstimator implements TopItems.Estimator<Comparable<?>> {

    private final List<Comparable<?>> toItemIDs;
    private final ItemSimilarity similarity;
    private final Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer;

    private MultiMostSimilarEstimator(List<Comparable<?>> toItemIDs,
                                      ItemSimilarity similarity,
                                      Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer) {
      this.toItemIDs = toItemIDs;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      RunningAverage average = new FullRunningAverage();
      for (Comparable<?> toItemID : toItemIDs) {
        Pair<Comparable<?>, Comparable<?>> pair = new Pair<Comparable<?>, Comparable<?>>(toItemID, itemID);
        if (rescorer != null && rescorer.isFiltered(pair)) {
          continue;
        }
        double estimate = similarity.itemSimilarity(toItemID, itemID);
        if (rescorer != null) {
          estimate = rescorer.rescore(pair, estimate);
        }
        average.addDatum(estimate);
      }
      return average.getAverage();
    }
  }

  private class RecommendedBecauseEstimator implements TopItems.Estimator<Comparable<?>> {

    private final Comparable<?> userID;
    private final Comparable<?> recommendedItemID;
    private final ItemSimilarity similarity;

    private RecommendedBecauseEstimator(Comparable<?> userID,
                                        Comparable<?> recommendedItemID,
                                        ItemSimilarity similarity) {
      this.userID = userID;
      this.recommendedItemID = recommendedItemID;
      this.similarity = similarity;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      Float pref = getDataModel().getPreferenceValue(userID, itemID);
      if (pref == null) {
        return Float.NaN;
      }
      double similarityValue = similarity.itemSimilarity(recommendedItemID, itemID);
      return (1.0 + similarityValue) * pref;
    }
  }

}
