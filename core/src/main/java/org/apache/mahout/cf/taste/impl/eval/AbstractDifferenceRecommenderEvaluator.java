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
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** <p>Abstract superclass of a couple implementations, providing shared functionality.</p> */
abstract class AbstractDifferenceRecommenderEvaluator implements RecommenderEvaluator {

  private static final Logger log = LoggerFactory.getLogger(AbstractDifferenceRecommenderEvaluator.class);

  private final Random random;

  AbstractDifferenceRecommenderEvaluator() {
    random = RandomUtils.getRandom();
  }

  @Override
  public double evaluate(RecommenderBuilder recommenderBuilder,
                         DataModelBuilder dataModelBuilder,
                         DataModel dataModel,
                         double trainingPercentage,
                         double evaluationPercentage) throws TasteException {

    if (recommenderBuilder == null) {
      throw new IllegalArgumentException("recommenderBuilder is null");
    }
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    if (Double.isNaN(trainingPercentage) || trainingPercentage <= 0.0 || trainingPercentage >= 1.0) {
      throw new IllegalArgumentException("Invalid trainingPercentage: " + trainingPercentage);
    }
    if (Double.isNaN(evaluationPercentage) || evaluationPercentage <= 0.0 || evaluationPercentage > 1.0) {
      throw new IllegalArgumentException("Invalid evaluationPercentage: " + evaluationPercentage);
    }

    log.info("Beginning evaluation using " + trainingPercentage + " of " + dataModel);

    int numUsers = dataModel.getNumUsers();
    FastByIDMap<PreferenceArray> trainingUsers =
            new FastByIDMap<PreferenceArray>(1 + (int) (evaluationPercentage * (double) numUsers));
    FastByIDMap<PreferenceArray> testUserPrefs =
            new FastByIDMap<PreferenceArray>(1 + (int) (evaluationPercentage * (double) numUsers));

    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.nextLong();
      if (random.nextDouble() < evaluationPercentage) {
        processOneUser(trainingPercentage, trainingUsers, testUserPrefs, userID, dataModel);
      }
    }

    DataModel trainingModel = dataModelBuilder == null ?
            new GenericDataModel(trainingUsers) :
            dataModelBuilder.buildDataModel(trainingUsers);

    Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

    log.info("Beginning evaluation for {} prefs", testUserPrefs.size());
    double result = getEvaluation(testUserPrefs, recommender);
    log.info("Evaluation result: " + result);
    return result;
  }

  private void processOneUser(double trainingPercentage,
                              FastByIDMap<PreferenceArray> trainingUsers,
                              FastByIDMap<PreferenceArray> testUserPrefs,
                              long userID,
                              DataModel dataModel) throws TasteException {
    List<Preference> trainingPrefs = null;
    List<Preference> testPrefs = null;
    PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      Preference newPref = new GenericPreference(userID, prefs.getItemID(i), prefs.getValue(i));
      if (random.nextDouble() < trainingPercentage) {
        if (trainingPrefs == null) {
          trainingPrefs = new ArrayList<Preference>(3);
        }
        trainingPrefs.add(newPref);
      } else {
        if (testPrefs == null) {
          testPrefs = new ArrayList<Preference>(3);
        }
        testPrefs.add(newPref);
      }
    }
    if (trainingPrefs != null) {
      trainingUsers.put(userID, new GenericUserPreferenceArray(trainingPrefs));
      if (testPrefs != null) {
        testUserPrefs.put(userID, new GenericUserPreferenceArray(testPrefs));
      }
    }
  }

  abstract double getEvaluation(FastByIDMap<PreferenceArray> testUserPrefs, Recommender recommender)
      throws TasteException;

}
