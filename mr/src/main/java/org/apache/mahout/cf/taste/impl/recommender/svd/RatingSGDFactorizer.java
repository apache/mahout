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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

/** Matrix factorization with user and item biases for rating prediction, trained with plain vanilla SGD  */
public class RatingSGDFactorizer extends AbstractFactorizer {

  protected static final int FEATURE_OFFSET = 3;
  
  /** Multiplicative decay factor for learning_rate */
  protected final double learningRateDecay;
  /** Learning rate (step size) */
  protected final double learningRate;
  /** Parameter used to prevent overfitting. */
  protected final double preventOverfitting;
  /** Number of features used to compute this factorization */
  protected final int numFeatures;
  /** Number of iterations */
  private final int numIterations;
  /** Standard deviation for random initialization of features */
  protected final double randomNoise;
  /** User features */
  protected double[][] userVectors;
  /** Item features */
  protected double[][] itemVectors;
  protected final DataModel dataModel;
  private long[] cachedUserIDs;
  private long[] cachedItemIDs;

  protected double biasLearningRate = 0.5;
  protected double biasReg = 0.1;

  /** place in user vector where the bias is stored */
  protected static final int USER_BIAS_INDEX = 1;
  /** place in item vector where the bias is stored */
  protected static final int ITEM_BIAS_INDEX = 2;

  public RatingSGDFactorizer(DataModel dataModel, int numFeatures, int numIterations) throws TasteException {
    this(dataModel, numFeatures, 0.01, 0.1, 0.01, numIterations, 1.0);
  }

  public RatingSGDFactorizer(DataModel dataModel, int numFeatures, double learningRate, double preventOverfitting,
      double randomNoise, int numIterations, double learningRateDecay) throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures + FEATURE_OFFSET;
    this.numIterations = numIterations;

    this.learningRate = learningRate;
    this.learningRateDecay = learningRateDecay;
    this.preventOverfitting = preventOverfitting;
    this.randomNoise = randomNoise;
  }

  protected void prepareTraining() throws TasteException {
    RandomWrapper random = RandomUtils.getRandom();
    userVectors = new double[dataModel.getNumUsers()][numFeatures];
    itemVectors = new double[dataModel.getNumItems()][numFeatures];

    double globalAverage = getAveragePreference();
    for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
      userVectors[userIndex][0] = globalAverage;
      userVectors[userIndex][USER_BIAS_INDEX] = 0; // will store user bias
      userVectors[userIndex][ITEM_BIAS_INDEX] = 1; // corresponding item feature contains item bias
      for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
        userVectors[userIndex][feature] = random.nextGaussian() * randomNoise;
      }
    }
    for (int itemIndex = 0; itemIndex < itemVectors.length; itemIndex++) {
      itemVectors[itemIndex][0] = 1; // corresponding user feature contains global average
      itemVectors[itemIndex][USER_BIAS_INDEX] = 1; // corresponding user feature contains user bias
      itemVectors[itemIndex][ITEM_BIAS_INDEX] = 0; // will store item bias
      for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
        itemVectors[itemIndex][feature] = random.nextGaussian() * randomNoise;
      }
    }

    cachePreferences();
    shufflePreferences();
  }

  private int countPreferences() throws TasteException {
    int numPreferences = 0;
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userIDs.nextLong());
      numPreferences += preferencesFromUser.length();
    }
    return numPreferences;
  }

  private void cachePreferences() throws TasteException {
    int numPreferences = countPreferences();
    cachedUserIDs = new long[numPreferences];
    cachedItemIDs = new long[numPreferences];

    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    int index = 0;
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userID);
      for (Preference preference : preferencesFromUser) {
        cachedUserIDs[index] = userID;
        cachedItemIDs[index] = preference.getItemID();
        index++;
      }
    }
  }

  protected void shufflePreferences() {
    RandomWrapper random = RandomUtils.getRandom();
    /* Durstenfeld shuffle */
    for (int currentPos = cachedUserIDs.length - 1; currentPos > 0; currentPos--) {
      int swapPos = random.nextInt(currentPos + 1);
      swapCachedPreferences(currentPos, swapPos);
    }
  }

  private void swapCachedPreferences(int posA, int posB) {
    long tmpUserIndex = cachedUserIDs[posA];
    long tmpItemIndex = cachedItemIDs[posA];

    cachedUserIDs[posA] = cachedUserIDs[posB];
    cachedItemIDs[posA] = cachedItemIDs[posB];

    cachedUserIDs[posB] = tmpUserIndex;
    cachedItemIDs[posB] = tmpItemIndex;
  }

  @Override
  public Factorization factorize() throws TasteException {
    prepareTraining();
    double currentLearningRate = learningRate;


    for (int it = 0; it < numIterations; it++) {
      for (int index = 0; index < cachedUserIDs.length; index++) {
        long userId = cachedUserIDs[index];
        long itemId = cachedItemIDs[index];
        float rating = dataModel.getPreferenceValue(userId, itemId);
        updateParameters(userId, itemId, rating, currentLearningRate);
      }
      currentLearningRate *= learningRateDecay;
    }
    return createFactorization(userVectors, itemVectors);
  }

  double getAveragePreference() throws TasteException {
    RunningAverage average = new FullRunningAverage();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
        average.addDatum(pref.getValue());
      }
    }
    return average.getAverage();
  }

  protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
    int userIndex = userIndex(userID);
    int itemIndex = itemIndex(itemID);

    double[] userVector = userVectors[userIndex];
    double[] itemVector = itemVectors[itemIndex];
    double prediction = predictRating(userIndex, itemIndex);
    double err = rating - prediction;

    // adjust user bias
    userVector[USER_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);

    // adjust item bias
    itemVector[ITEM_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);

    // adjust features
    for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
      double userFeature = userVector[feature];
      double itemFeature = itemVector[feature];

      double deltaUserFeature = err * itemFeature - preventOverfitting * userFeature;
      userVector[feature] += currentLearningRate * deltaUserFeature;

      double deltaItemFeature = err * userFeature - preventOverfitting * itemFeature;
      itemVector[feature] += currentLearningRate * deltaItemFeature;
    }
  }

  private double predictRating(int userID, int itemID) {
    double sum = 0;
    for (int feature = 0; feature < numFeatures; feature++) {
      sum += userVectors[userID][feature] * itemVectors[itemID][feature];
    }
    return sum;
  }
}
