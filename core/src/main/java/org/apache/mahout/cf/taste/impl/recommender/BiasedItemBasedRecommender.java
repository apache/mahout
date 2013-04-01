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

import com.google.common.primitives.Doubles;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.math.Sorting;
import org.apache.mahout.math.Swapper;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.map.OpenLongDoubleHashMap;

/**
 * item-based recommender that uses weighted sum estimation enhanced by baseline estimates, porting baseline estimation
 * from the "UserItemBaseline" rating predictor from "mymedialite" https://github.com/zenogantner/MyMediaLite/
 */
public class BiasedItemBasedRecommender extends GenericItemBasedRecommender {
  
  private final int numSimilarItems;
  
  private final double averageRating;
  private final OpenLongDoubleHashMap itemBiases;
  private final OpenLongDoubleHashMap userBiases;

  private static final int DEFAULT_NUM_SIMILAR_ITEMS = 50;
  private static final int DEFAULT_NUM_OPTIMIZATION_PASSES = 5;
  private static final double DEFAULT_USER_BIAS_REGULARIZATION = 10;
  private static final double DEFAULT_ITEM_BIAS_REGULARIZATION = 5;

  private final ItemSimilarity similarity;

  public BiasedItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity) throws TasteException {
    this(dataModel, similarity, DEFAULT_NUM_SIMILAR_ITEMS, DEFAULT_NUM_OPTIMIZATION_PASSES,
        DEFAULT_ITEM_BIAS_REGULARIZATION, DEFAULT_USER_BIAS_REGULARIZATION);
  }

  public BiasedItemBasedRecommender(DataModel dataModel, ItemSimilarity similarity, int numSimilarItems,
      int numOptimizationPasses, double itemBiasRegularization, double userBiasRegularization) throws TasteException {
    super(dataModel, similarity);
    this.numSimilarItems = numSimilarItems;
    this.similarity = similarity;

    averageRating = averageRating();

    itemBiases = new OpenLongDoubleHashMap(getDataModel().getNumItems());
    userBiases = new OpenLongDoubleHashMap(getDataModel().getNumUsers());

    for (int pass = 0; pass < numOptimizationPasses; pass++) {
      optimizeItemBiases(itemBiasRegularization);
      optimizeUserBiases(userBiasRegularization);
    }
  }

  private void optimizeItemBiases(double itemBiasRegularization) throws TasteException {
    LongPrimitiveIterator itemIDs = getDataModel().getItemIDs();
    while (itemIDs.hasNext()) {
      long itemID = itemIDs.nextLong();
      PreferenceArray preferences = getDataModel().getPreferencesForItem(itemID);
      double sum = 0;
      for (Preference pref : preferences) {
        sum += pref.getValue() - averageRating;
      }
      double bias = sum / (itemBiasRegularization + preferences.length());
      itemBiases.put(itemID, bias);
    }
  }

  private void optimizeUserBiases(double userBiasRegularization) throws TasteException {
    LongPrimitiveIterator userIDs = getDataModel().getUserIDs();
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      PreferenceArray preferences = getDataModel().getPreferencesFromUser(userID);
      double sum = 0;
      for (Preference pref : preferences) {
        sum += pref.getValue() - averageRating - itemBiases.get(pref.getItemID());
      }
      double bias = sum / (userBiasRegularization + preferences.length());
      userBiases.put(userID, bias);
    }
  }

  private double averageRating() throws TasteException {
    RunningAverage averageRating = new FullRunningAverage();
    LongPrimitiveIterator itemIDs = getDataModel().getItemIDs();
    while (itemIDs.hasNext()) {
      for (Preference pref : getDataModel().getPreferencesForItem(itemIDs.next())) {
        averageRating.addDatum(pref.getValue());
      }
    }
    return averageRating.getAverage();
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
    Float actualPref = getPreferenceForItem(preferencesFromUser, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    return doEstimatePreference(userID, preferencesFromUser, itemID);
  }

  private static Float getPreferenceForItem(PreferenceArray preferencesFromUser, long itemID) {
    int size = preferencesFromUser.length();
    for (int i = 0; i < size; i++) {
      if (preferencesFromUser.getItemID(i) == itemID) {
        return preferencesFromUser.getValue(i);
      }
    }
    return null;
  }

  protected double baselineEstimate(long userID, long itemID) {
    return averageRating + userBiases.get(userID) + itemBiases.get(itemID);
  }

  @Override
  protected float doEstimatePreference(long userID, PreferenceArray preferencesFromUser, long itemID)
    throws TasteException {
    long[] userIDs = preferencesFromUser.getIDs();
    float[] ratings = new float[userIDs.length];
    long[] itemIDs = new long[userIDs.length];
            
    final double[] similarities = similarity.itemSimilarities(itemID, userIDs);

    for (int n = 0; n < preferencesFromUser.length(); n++) {
      ratings[n] = preferencesFromUser.get(n).getValue();
      itemIDs[n] = preferencesFromUser.get(n).getItemID();
    }

    // sort, so that we can only use the top similarities
    Sorting.quickSort(0, similarities.length, new SimilaritiesComparator(similarities),
        new SimilaritiesRatingsItemIDsSwapper(similarities, ratings, itemIDs));

    double preference = 0.0;
    double totalSimilarity = 0.0;
    int count = 0;
    for (int i = 0; i < Math.min(numSimilarItems, similarities.length); i++) {
      double theSimilarity = similarities[i];
      if (!Double.isNaN(theSimilarity)) {
        preference += theSimilarity * (ratings[i] - baselineEstimate(userID, itemIDs[i]));
        totalSimilarity += Math.abs(theSimilarity);
        count++;
      }
    }

    if (count <= 1) {
      return Float.NaN;
    }

    return (float) (baselineEstimate(userID, itemID) + (preference / totalSimilarity));
  }

  static class SimilaritiesComparator implements IntComparator {

    private final double[] similarities;

    SimilaritiesComparator(double[] similarities) {
      this.similarities = similarities;
    }

    @Override
    public int compare(int pos1, int pos2) {
      return -1 * Doubles.compare(similarities[pos1], similarities[pos2]);
    }
  }

  static class SimilaritiesRatingsItemIDsSwapper implements Swapper {

    private final double[] similarities;
    private final float[] ratings;
    private final long[] itemIDs;

    SimilaritiesRatingsItemIDsSwapper(double[] similarities, float[] ratings, long[] itemIDs) {
      this.similarities = similarities;
      this.ratings = ratings;
      this.itemIDs = itemIDs;
    }

    @Override
    public void swap(int a, int b) {
      double tempDouble = similarities[b];
      similarities[b] = similarities[a];
      similarities[a] = tempDouble;

      float tempFloat = ratings[b];
      ratings[b] = ratings[a];
      ratings[a] = tempFloat;

      long tempLong = itemIDs[b];
      itemIDs[b] = itemIDs[a];
      itemIDs[a] = tempLong;
    }
  }

}
