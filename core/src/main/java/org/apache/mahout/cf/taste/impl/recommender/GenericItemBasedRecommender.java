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
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPair;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
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
  public List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer)
      throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    if (getNumPreferences(userID) == 0) {
      return Collections.emptyList();
    }

    FastIDSet allItemIDs = getAllOtherItems(userID);

    TopItems.Estimator<Long> estimator = new Estimator(userID);

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
    return doEstimatePreference(userID, itemID);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(long itemID, int howMany) throws TasteException {
    return mostSimilarItems(itemID, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(long itemID,
                                                int howMany,
                                                Rescorer<LongPair> rescorer) throws TasteException {
    TopItems.Estimator<Long> estimator = new MostSimilarEstimator(itemID, similarity, rescorer);
    return doMostSimilarItems(itemID, howMany, estimator);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany) throws TasteException {
    return mostSimilarItems(itemIDs, howMany, null);
  }

  @Override
  public List<RecommendedItem> mostSimilarItems(long[] itemIDs,
                                                int howMany,
                                                Rescorer<LongPair> rescorer) throws TasteException {
    DataModel model = getDataModel();
    TopItems.Estimator<Long> estimator = new MultiMostSimilarEstimator(itemIDs, similarity, rescorer);
    FastIDSet allItemIDs = new FastIDSet(model.getNumItems());
    LongPrimitiveIterator it = model.getItemIDs();
    while (it.hasNext()) {
      allItemIDs.add(it.nextLong());
    }
    allItemIDs.removeAll(itemIDs);
    return TopItems.getTopItems(howMany, allItemIDs.iterator(), null, estimator);
  }

  @Override
  public List<RecommendedItem> recommendedBecause(long userID,
                                                  long itemID,
                                                  int howMany) throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    DataModel model = getDataModel();
    TopItems.Estimator<Long> estimator = new RecommendedBecauseEstimator(userID, itemID, similarity);

    PreferenceArray prefs = model.getPreferencesFromUser(userID);
    int size = prefs.length();
    FastIDSet allUserItems = new FastIDSet(size);    
    for (int i = 0; i < size; i++) {
      allUserItems.add(prefs.getItemID(i));
    }
    allUserItems.remove(itemID);

    return TopItems.getTopItems(howMany, allUserItems.iterator(), null, estimator);
  }

  private List<RecommendedItem> doMostSimilarItems(long itemID,
                                                   int howMany,
                                                   TopItems.Estimator<Long> estimator) throws TasteException {
    DataModel model = getDataModel();
    FastIDSet allItemIDs = new FastIDSet(model.getNumItems());
    LongPrimitiveIterator it = model.getItemIDs();
    while (it.hasNext()) {
      allItemIDs.add(it.nextLong());
    }
    allItemIDs.remove(itemID);
    return TopItems.getTopItems(howMany, allItemIDs.iterator(), null, estimator);
  }

  protected float doEstimatePreference(long userID, long itemID) throws TasteException {
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

  private int getNumPreferences(long userID) throws TasteException {
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

  public static class MostSimilarEstimator implements TopItems.Estimator<Long> {

    private final long toItemID;
    private final ItemSimilarity similarity;
    private final Rescorer<LongPair> rescorer;

    public MostSimilarEstimator(long toItemID,
                                ItemSimilarity similarity,
                                Rescorer<LongPair> rescorer) {
      this.toItemID = toItemID;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      LongPair pair = new LongPair(toItemID, itemID);
      if (rescorer != null && rescorer.isFiltered(pair)) {
        return Double.NaN;
      }
      double originalEstimate = similarity.itemSimilarity(toItemID, itemID);
      return rescorer == null ? originalEstimate : rescorer.rescore(pair, originalEstimate);
    }
  }

  private final class Estimator implements TopItems.Estimator<Long> {

    private final long userID;

    private Estimator(long userID) {
      this.userID = userID;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return doEstimatePreference(userID, itemID);
    }
  }

  private static class MultiMostSimilarEstimator implements TopItems.Estimator<Long> {

    private final long[] toItemIDs;
    private final ItemSimilarity similarity;
    private final Rescorer<LongPair> rescorer;

    private MultiMostSimilarEstimator(long[] toItemIDs,
                                      ItemSimilarity similarity,
                                      Rescorer<LongPair> rescorer) {
      this.toItemIDs = toItemIDs;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      RunningAverage average = new FullRunningAverage();
      for (long toItemID : toItemIDs) {
        LongPair pair = new LongPair(toItemID, itemID);
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

  private class RecommendedBecauseEstimator implements TopItems.Estimator<Long> {

    private final long userID;
    private final long recommendedItemID;
    private final ItemSimilarity similarity;

    private RecommendedBecauseEstimator(long userID,
                                        long recommendedItemID,
                                        ItemSimilarity similarity) {
      this.userID = userID;
      this.recommendedItemID = recommendedItemID;
      this.similarity = similarity;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      Float pref = getDataModel().getPreferenceValue(userID, itemID);
      if (pref == null) {
        return Float.NaN;
      }
      double similarityValue = similarity.itemSimilarity(recommendedItemID, itemID);
      return (1.0 + similarityValue) * pref;
    }
  }

}
