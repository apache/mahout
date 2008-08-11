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
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.Random;
import java.util.List;

/**
 * <p>Abstract superclass of a couple implementations, providing shared functionality.</p>
 */
abstract class AbstractDifferenceRecommenderEvaluator implements RecommenderEvaluator {

  private static final Logger log = LoggerFactory.getLogger(AbstractDifferenceRecommenderEvaluator.class);

  private final Random random;

  AbstractDifferenceRecommenderEvaluator() {
    random = RandomUtils.getRandom();
  }

  public double evaluate(RecommenderBuilder recommenderBuilder,
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
    Collection<User> trainingUsers = new ArrayList<User>(1 + (int) (trainingPercentage * (double) numUsers));
    Map<User, Collection<Preference>> testUserPrefs =
            new FastMap<User, Collection<Preference>>(1 + (int) ((1.0 - trainingPercentage) * (double) numUsers));

    for (User user : dataModel.getUsers()) {
      if (random.nextDouble() < evaluationPercentage) {
        processOneUser(trainingPercentage, trainingUsers, testUserPrefs, user);
      }
    }

    DataModel trainingModel = new GenericDataModel(trainingUsers);

    Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);

    double result = getEvaluation(testUserPrefs, recommender);
    log.info("Evaluation result: " + result);
    return result;
  }

  private void processOneUser(double trainingPercentage,
                              Collection<User> trainingUsers,
                              Map<User, Collection<Preference>> testUserPrefs,
                              User user) {
    List<Preference> trainingPrefs = new ArrayList<Preference>();
    List<Preference> testPrefs = new ArrayList<Preference>();
    Preference[] prefs = user.getPreferencesAsArray();
    for (int i = 0; i < prefs.length; i++) {
      Preference pref = prefs[i];
      Item itemCopy = new GenericItem<String>(pref.getItem().getID().toString());
      Preference newPref = new GenericPreference(null, itemCopy, pref.getValue());
      if (random.nextDouble() < trainingPercentage) {
        trainingPrefs.add(newPref);
      } else {
        testPrefs.add(newPref);
      }
    }
    log.debug("Training against {} preferences", trainingPrefs.size());
    log.debug("Evaluating accuracy of {} preferences", testPrefs.size());
    if (!trainingPrefs.isEmpty()) {
      User trainingUser = new GenericUser<String>(user.getID().toString(), trainingPrefs);
      trainingUsers.add(trainingUser);
      if (!testPrefs.isEmpty()) {
        testUserPrefs.put(trainingUser, testPrefs);
      }
    }
  }

  abstract double getEvaluation(Map<User, Collection<Preference>> testUserPrefs, Recommender recommender)
      throws TasteException;

}
