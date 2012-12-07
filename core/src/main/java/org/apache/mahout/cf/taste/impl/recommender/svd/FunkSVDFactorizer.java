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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of Simon Funk's famous algorithm from the Netflix prize,,
 * see http://sifter.org/~simon/journal/20061211.html for details
 */
@Deprecated
public final class FunkSVDFactorizer extends AbstractFactorizer {

  private static final Logger log = LoggerFactory.getLogger(FunkSVDFactorizer.class);

  private final double learningRate;
  /** used to prevent overfitting.*/
  private final double regularization;
  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** number of iterations */
  private final int numIterations;
  private final double randomNoise;
  private double[][] userVectors;
  private double[][] itemVectors;
  private final DataModel dataModel;
  private List<SVDPreference> cachedPreferences;
  private double defaultValue;
  private double interval;

  private static final double DEFAULT_LEARNING_RATE = 0.005;
  private static final double DEFAULT_REGULARIZATION = 0.02;
  private static final double DEFAULT_RANDOM_NOISE = 0.005;

  public FunkSVDFactorizer(DataModel dataModel, int numFeatures, int numIterations) throws TasteException {
    this(dataModel, numFeatures, DEFAULT_LEARNING_RATE, DEFAULT_REGULARIZATION, DEFAULT_RANDOM_NOISE,
        numIterations);
  }

  public FunkSVDFactorizer(DataModel dataModel, int numFeatures, double learningRate, double regularization,
      double randomNoise, int numIterations) throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.numIterations = numIterations;

    this.learningRate = learningRate;
    this.regularization = regularization;
    this.randomNoise = randomNoise;
  }

  @Override
  public Factorization factorize() throws TasteException {
    Random random = RandomUtils.getRandom();
    userVectors = new double[dataModel.getNumUsers()][numFeatures];
    itemVectors = new double[dataModel.getNumItems()][numFeatures];

    double average = getAveragePreference();

    double prefInterval = dataModel.getMaxPreference() - dataModel.getMinPreference();
    defaultValue = Math.sqrt((average - prefInterval * 0.1) / numFeatures);
    interval = prefInterval * 0.1 / numFeatures;

    for (int feature = 0; feature < numFeatures; feature++) {
      for (int userIndex = 0; userIndex < dataModel.getNumUsers(); userIndex++) {
        userVectors[userIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
      for (int itemIndex = 0; itemIndex < dataModel.getNumItems(); itemIndex++) {
        itemVectors[itemIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
    }
    cachedPreferences = Lists.newArrayListWithCapacity(dataModel.getNumUsers());
    cachePreferences();
    double rmse = dataModel.getMaxPreference() - dataModel.getMinPreference();
    for (int feature = 0; feature < numFeatures; feature++) {
      Collections.shuffle(cachedPreferences, random);
      for (int i = 0; i < numIterations; i++) {
        double err = 0.0;
        for (SVDPreference pref : cachedPreferences) {
          int useridx = userIndex(pref.getUserID());
          int itemidx = itemIndex(pref.getItemID());
          err += Math.pow(train(useridx, itemidx, feature, pref), 2.0);
        }
        rmse = Math.sqrt(err / cachedPreferences.size());
      }
      if (feature < numFeatures - 1) {
        for (SVDPreference preference : cachedPreferences) {
          int useridx = userIndex(preference.getUserID());
          int itemidx = itemIndex(preference.getItemID());
          buildCache(useridx, itemidx, feature, preference);
        }
      }
      log.info("Finished training feature {} with RMSE {}.", feature, rmse);
    }
    return createFactorization(userVectors, itemVectors);
  }

  double getAveragePreference() throws TasteException {
    RunningAverage average = new FullRunningAverage();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      for (Preference preference : dataModel.getPreferencesFromUser(userIDs.nextLong())) {
        average.addDatum(preference.getValue());
      }
    }
    return average.getAverage();
  }

  private double train(int userIndex, int itemIndex, int feature, SVDPreference pref) {
    double[] userVector = userVectors[userIndex];
    double[] itemVector = itemVectors[itemIndex];
    double prediction = predictRating(userIndex, itemIndex, feature, pref, true);
    double err = pref.getValue() - prediction;
    userVector[feature] += learningRate * (err * itemVector[feature] - regularization * userVector[feature]);
    itemVector[feature] += learningRate * (err * userVector[feature] - regularization * itemVector[feature]);
    return err;
  }

  private void buildCache(int userIndex, int itemIndex, int k, SVDPreference pref) {
    pref.setCache(predictRating(userIndex, itemIndex, k, pref, false));
  }

  private double predictRating(int userIndex, int itemIndex, int feature, SVDPreference pref, boolean trailing) {
    float minPreference = dataModel.getMinPreference();
    float maxPreference = dataModel.getMaxPreference();
    double sum = pref.getCache();
    sum += userVectors[userIndex][feature] * itemVectors[itemIndex][feature];
    if (trailing) {
      sum += (numFeatures - feature - 1) * (defaultValue + interval) * (defaultValue + interval);
      if (sum > maxPreference) {
        sum = maxPreference;
      } else if (sum < minPreference) {
        sum = minPreference;
      }
    }
    return sum;
  }

  private void cachePreferences() throws TasteException {
    cachedPreferences.clear();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      for (Preference pref : dataModel.getPreferencesFromUser(userIDs.nextLong())) {
        cachedPreferences.add(new SVDPreference(pref.getUserID(), pref.getItemID(), pref.getValue(), 0.0));
      }
    }
  }

}
