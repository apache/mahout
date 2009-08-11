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

package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * <p>For each user, these implementation determine the top <code>n</code> preferences, then evaluate the IR
 * statistics based on a {@link DataModel} that does not have these values. This number <code>n</code> is the "at"
 * value, as in "precision at 5". For example, this would mean precision evaluated by removing the top 5 preferences for
 * a user and then finding the percentage of those 5 items included in the top 5 recommendations for
 * that user.</p>
 */
public final class GenericRecommenderIRStatsEvaluator implements RecommenderIRStatsEvaluator {

  private static final Logger log = LoggerFactory.getLogger(GenericRecommenderIRStatsEvaluator.class);

  /**
   * Pass as "relevanceThreshold" argument to
   * {@link #evaluate(RecommenderBuilder, DataModelBuilder, DataModel, Rescorer, int, double, double)}
   * to have it attempt to compute a reasonable threshold. Note that this will impact performance.
   */
  public static final double CHOOSE_THRESHOLD = Double.NaN;

  private final Random random;

  public GenericRecommenderIRStatsEvaluator() {
    random = RandomUtils.getRandom();
  }

  @Override
  public IRStatistics evaluate(RecommenderBuilder recommenderBuilder,
                               DataModelBuilder dataModelBuilder,
                               DataModel dataModel,
                               Rescorer<Long> rescorer,
                               int at,
                               double relevanceThreshold,
                               double evaluationPercentage) throws TasteException {

    if (recommenderBuilder == null) {
      throw new IllegalArgumentException("recommenderBuilder is null");
    }
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    if (at < 1) {
      throw new IllegalArgumentException("at must be at least 1");
    }
    if (Double.isNaN(evaluationPercentage) || evaluationPercentage <= 0.0 || evaluationPercentage > 1.0) {
      throw new IllegalArgumentException("Invalid evaluationPercentage: " + evaluationPercentage);
    }
    if (Double.isNaN(relevanceThreshold)) {
      throw new IllegalArgumentException("Invalid relevanceThreshold: " + evaluationPercentage);
    }

    int numItems = dataModel.getNumItems();
    RunningAverage precision = new FullRunningAverage();
    RunningAverage recall = new FullRunningAverage();
    RunningAverage fallOut = new FullRunningAverage();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.nextLong();
      if (random.nextDouble() < evaluationPercentage) {
        long start = System.currentTimeMillis();
        FastIDSet relevantItemIDs = new FastIDSet(at);
        PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
        int size = prefs.length();
        if (size < 2 * at) {
          // Really not enough prefs to meaningfully evaluate this user
          continue;
        }

        // List some most-preferred items that would count as (most) "relevant" results
        double theRelevanceThreshold = Double.isNaN(relevanceThreshold) ? computeThreshold(prefs) : relevanceThreshold;
        prefs.sortByValueReversed();
        for (int i = 0; i < size && relevantItemIDs.size() < at; i++) {
          if (prefs.getValue(i) >= theRelevanceThreshold) {
            relevantItemIDs.add(prefs.getItemID(i));
          }
        }
        int numRelevantItems = relevantItemIDs.size();
        if (numRelevantItems > 0) {
          FastByIDMap<PreferenceArray> trainingUsers =
            new FastByIDMap<PreferenceArray>(dataModel.getNumUsers());
          LongPrimitiveIterator it2 = dataModel.getUserIDs();
          while (it2.hasNext()) {
            processOtherUser(userID, relevantItemIDs, trainingUsers, it2.nextLong(), dataModel);
          }

          DataModel trainingModel = dataModelBuilder == null ?
            new GenericDataModel(trainingUsers) :
            dataModelBuilder.buildDataModel(trainingUsers);
          Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

          try {
            trainingModel.getPreferencesFromUser(userID);
          } catch (NoSuchUserException nsee) {
            continue; // Oops we excluded all prefs for the user -- just move on
          }

          int intersectionSize = 0;
          List<RecommendedItem> recommendedItems = recommender.recommend(userID, at, rescorer);
          for (RecommendedItem recommendedItem : recommendedItems) {
            if (relevantItemIDs.contains(recommendedItem.getItemID())) {
              intersectionSize++;
            }
          }
          int numRecommendedItems = recommendedItems.size();
          if (numRecommendedItems > 0) {
            precision.addDatum((double) intersectionSize / (double) numRecommendedItems);
          }
          recall.addDatum((double) intersectionSize / (double) numRelevantItems);
          if (numRelevantItems < size) {
            fallOut.addDatum((double) (numRecommendedItems - intersectionSize) /
                (double) (numItems - numRelevantItems));
          }

          long end = System.currentTimeMillis();
          log.info("Evaluated with user " + userID + " in " + (end - start) + "ms");
          log.info("Precision/recall/fall-out: {} / {} / {}", new Object[]{
              precision.getAverage(), recall.getAverage(), fallOut.getAverage()
          });
        }
      }
    }

    return new IRStatisticsImpl(precision.getAverage(), recall.getAverage(), fallOut.getAverage());
  }

  private static void processOtherUser(long id,
                                       FastIDSet relevantItemIDs,
                                       FastByIDMap<PreferenceArray> trainingUsers,
                                       long userID2,
                                       DataModel dataModel) throws TasteException {
    PreferenceArray prefs2Array = dataModel.getPreferencesFromUser(userID2);
    if (id == userID2) {
      List<Preference> prefs2 = new ArrayList<Preference>(prefs2Array.length());
      for (Preference pref : prefs2Array) {
        prefs2.add(pref);
      }
      for (Iterator<Preference> iterator = prefs2.iterator(); iterator.hasNext();) {
        Preference pref = iterator.next();
        if (relevantItemIDs.contains(pref.getItemID())) {
          iterator.remove();
        }
      }
      prefs2Array = new GenericUserPreferenceArray(prefs2);
    }
    trainingUsers.put(userID2, prefs2Array);
  }

  private static double computeThreshold(PreferenceArray prefs) {
    if (prefs.length() < 2) {
      // Not enough data points -- return a threshold that allows everything
      return Double.NEGATIVE_INFINITY;
    }
    RunningAverageAndStdDev stdDev = new FullRunningAverageAndStdDev();
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      stdDev.addDatum(prefs.getValue(i));
    }
    return stdDev.getAverage() + stdDev.getStandardDeviation();
  }

}
