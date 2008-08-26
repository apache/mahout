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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * <p>For each {@link User}, these implementation determine the top <code>n</code> preferences,
 * then evaluate the IR statistics based on a {@link DataModel} that does not have these values.
 * This number <code>n</code> is the "at" value, as in "precision at 5". For example, this would mean precision
 * evaluated by removing the top 5 preferences for a {@link User} and then finding the percentage of those 5
 * {@link Item}s included in the top 5 recommendations for that user.</p>
 */
public final class GenericRecommenderIRStatsEvaluator implements RecommenderIRStatsEvaluator {

  private static final Logger log = LoggerFactory.getLogger(GenericRecommenderIRStatsEvaluator.class);

  private final Random random;

  public GenericRecommenderIRStatsEvaluator() {
    random = RandomUtils.getRandom();
  }

  public IRStatistics evaluate(RecommenderBuilder recommenderBuilder,
                               DataModel dataModel,
                               Rescorer<Item> rescorer,
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

    RunningAverage precision = new FullRunningAverage();
    RunningAverage recall = new FullRunningAverage();
    RunningAverage fallOut = new FullRunningAverage();
    for (User user : dataModel.getUsers()) {
      if (random.nextDouble() < evaluationPercentage) {
        Object id = user.getID();
        Collection<Item> relevantItems = new HashSet<Item>(at);
        Preference[] prefs = user.getPreferencesAsArray();
        for (int i = 0; i < prefs.length; i++) {
          Preference pref = prefs[i];
          if (pref.getValue() >= relevanceThreshold) {
            relevantItems.add(pref.getItem());
          }
        }
        int numRelevantItems = relevantItems.size();
        if (numRelevantItems > 0) {
          List<User> trainingUsers = new ArrayList<User>(dataModel.getNumUsers());
          for (User user2 : dataModel.getUsers()) {
            processOtherUser(id, relevantItems, trainingUsers, user2);
          }
          DataModel trainingModel = new GenericDataModel(trainingUsers);
          Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

          try {
            trainingModel.getUser(id);
          } catch (NoSuchElementException nsee) {
            continue; // Oops we excluded all prefs for the user -- just move on
          }

          int intersectionSize = 0;
          List<RecommendedItem> recommendedItems;
          if (rescorer == null) {
            recommendedItems = recommender.recommend(id, at);
          } else {
            recommendedItems = recommender.recommend(id, at, rescorer);
          }
          for (RecommendedItem recommendedItem : recommendedItems) {
            if (relevantItems.contains(recommendedItem.getItem())) {
              intersectionSize++;
            }
          }
          int numRecommendedItems = recommendedItems.size();
          if (numRecommendedItems > 0) {
            precision.addDatum((double) intersectionSize / (double) numRecommendedItems);
          }
          recall.addDatum((double) intersectionSize / (double) numRelevantItems);
          if (numRelevantItems < prefs.length) {
            fallOut.addDatum((double) (numRecommendedItems - intersectionSize) /
                             (double) (prefs.length - numRelevantItems));
          }

          log.info("Precision/recall/fall-out: {} / {} / {}", new Object[] {
              precision.getAverage(), recall.getAverage(), fallOut.getAverage()
          });
        }
      }
    }

    return new IRStatisticsImpl(precision.getAverage(), recall.getAverage(), fallOut.getAverage());
  }

  private void processOtherUser(Object id,
                                Collection<Item> relevantItems,
                                Collection<User> trainingUsers,
                                User user2) {
    if (id.equals(user2.getID())) {
      List<Preference> trainingPrefs = new ArrayList<Preference>();
      Preference[] prefs2 = user2.getPreferencesAsArray();
      for (int i = 0; i < prefs2.length; i++) {
        Preference pref = prefs2[i];
        if (!relevantItems.contains(pref.getItem())) {
          trainingPrefs.add(pref);
        }
      }
      if (!trainingPrefs.isEmpty()) {
        User trainingUser = new GenericUser<String>(id.toString(), trainingPrefs);
        trainingUsers.add(trainingUser);
      }
    } else {
      trainingUsers.add(user2);
    }
  }

}
