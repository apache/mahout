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

package org.apache.mahout.cf.taste.example.kddcup.track1.svd;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Random;

/**
 * {@link Factorizer} based on Simon Funk's famous article <a href="http://sifter.org/~simon/journal/20061211.html">
 * "Netflix Update: Try this at home"</a>.
 *
 * Attempts to be as memory efficient as possible, only iterating once through the
 * {@link FactorizablePreferences} or {@link DataModel} while copying everything to primitive arrays.
 * Learning works in place on these datastructures after that.
 */
public class ParallelArraysSGDFactorizer implements Factorizer {

  public static final double DEFAULT_LEARNING_RATE = 0.005;
  public static final double DEFAULT_PREVENT_OVERFITTING = 0.02;
  public static final double DEFAULT_RANDOM_NOISE = 0.005;

  private final int numFeatures;
  private final int numIterations;
  private final float minPreference;
  private final float maxPreference;

  private final Random random;
  private final double learningRate;
  private final double preventOverfitting;

  private final FastByIDMap<Integer> userIDMapping;
  private final FastByIDMap<Integer> itemIDMapping;

  private final double[][] userFeatures;
  private final double[][] itemFeatures;

  private final int[] userIndexes;
  private final int[] itemIndexes;
  private final float[] values;

  private final double defaultValue;
  private final double interval;
  private final double[] cachedEstimates;


  private static final Logger log = LoggerFactory.getLogger(ParallelArraysSGDFactorizer.class);

  public ParallelArraysSGDFactorizer(DataModel dataModel, int numFeatures, int numIterations) {
    this(new DataModelFactorizablePreferences(dataModel), numFeatures, numIterations, DEFAULT_LEARNING_RATE,
        DEFAULT_PREVENT_OVERFITTING, DEFAULT_RANDOM_NOISE);
  }

  public ParallelArraysSGDFactorizer(DataModel dataModel, int numFeatures, int numIterations, double learningRate,
                                     double preventOverfitting, double randomNoise) {
    this(new DataModelFactorizablePreferences(dataModel), numFeatures, numIterations, learningRate, preventOverfitting,
        randomNoise);
  }

  public ParallelArraysSGDFactorizer(FactorizablePreferences factorizablePrefs, int numFeatures, int numIterations) {
    this(factorizablePrefs, numFeatures, numIterations, DEFAULT_LEARNING_RATE, DEFAULT_PREVENT_OVERFITTING,
        DEFAULT_RANDOM_NOISE);
  }

  public ParallelArraysSGDFactorizer(FactorizablePreferences factorizablePreferences, int numFeatures,
      int numIterations, double learningRate, double preventOverfitting, double randomNoise) {

    this.numFeatures = numFeatures;
    this.numIterations = numIterations;
    minPreference = factorizablePreferences.getMinPreference();
    maxPreference = factorizablePreferences.getMaxPreference();

    this.random = RandomUtils.getRandom();
    this.learningRate = learningRate;
    this.preventOverfitting = preventOverfitting;

    int numUsers = factorizablePreferences.numUsers();
    int numItems = factorizablePreferences.numItems();
    int numPrefs = factorizablePreferences.numPreferences();

    log.info("Mapping {} users...", numUsers);
    userIDMapping = new FastByIDMap<Integer>(numUsers);
    int index = 0;
    LongPrimitiveIterator userIterator = factorizablePreferences.getUserIDs();
    while (userIterator.hasNext()) {
      userIDMapping.put(userIterator.nextLong(), index++);
    }

    log.info("Mapping {} items", numItems);
    itemIDMapping = new FastByIDMap<Integer>(numItems);
    index = 0;
    LongPrimitiveIterator itemIterator = factorizablePreferences.getItemIDs();
    while (itemIterator.hasNext()) {
      itemIDMapping.put(itemIterator.nextLong(), index++);
    }

    this.userIndexes = new int[numPrefs];
    this.itemIndexes = new int[numPrefs];
    this.values = new float[numPrefs];
    this.cachedEstimates = new double[numPrefs];

    index = 0;
    log.info("Loading {} preferences into memory", numPrefs);
    RunningAverage average = new FullRunningAverage();
    for (Preference preference : factorizablePreferences.getPreferences()) {
      userIndexes[index] = userIDMapping.get(preference.getUserID());
      itemIndexes[index] = itemIDMapping.get(preference.getItemID());
      values[index] = preference.getValue();
      cachedEstimates[index] = 0;

      average.addDatum(preference.getValue());

      index++;
      if (index % 1000000 == 0) {
        log.info("Processed {} preferences", index);
      }
    }
    log.info("Processed {} preferences, done.", index);

    double averagePreference = average.getAverage();
    log.info("Average preference value is {}", averagePreference);

    double prefInterval = factorizablePreferences.getMaxPreference() - factorizablePreferences.getMinPreference();
    defaultValue = Math.sqrt((averagePreference - prefInterval * 0.1) / numFeatures);
    interval = prefInterval * 0.1 / numFeatures;

    userFeatures = new double[numUsers][numFeatures];
    itemFeatures = new double[numItems][numFeatures];

    log.info("Initializing feature vectors...");
    for (int feature = 0; feature < numFeatures; feature++) {
      for (int userIndex = 0; userIndex < numUsers; userIndex++) {
        userFeatures[userIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
      for (int itemIndex = 0; itemIndex < numItems; itemIndex++) {
        itemFeatures[itemIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
    }
  }

  @Override
  public Factorization factorize() throws TasteException {
    for (int feature = 0; feature < numFeatures; feature++) {
      log.info("Shuffling preferences...");
      shufflePreferences();
      log.info("Starting training of feature {} ...", feature);
      for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
        if (currentIteration == numIterations - 1) {
          double rmse = trainingIterationWithRmse(feature);
          log.info("Finished training feature {} with RMSE {}", feature, rmse);
        } else {
          trainingIteration(feature);
        }
      }
      if (feature < numFeatures - 1) {
        log.info("Updating cache...");
        for (int index = 0; index < userIndexes.length; index++) {
          cachedEstimates[index] = estimate(userIndexes[index], itemIndexes[index], feature, cachedEstimates[index],
              false);
        }
      }
    }
    log.info("Factorization done");
    return new Factorization(userIDMapping, itemIDMapping, userFeatures, itemFeatures);
  }

  private void trainingIteration(int feature) {
    for (int index = 0; index < userIndexes.length; index++) {
      train(userIndexes[index], itemIndexes[index], feature, values[index], cachedEstimates[index]);
    }
  }

  private double trainingIterationWithRmse(int feature) {
    double rmse = 0.0;
    for (int index = 0; index < userIndexes.length; index++) {
      double error = train(userIndexes[index], itemIndexes[index], feature, values[index], cachedEstimates[index]);
      rmse += error * error;
    }
    return Math.sqrt(rmse / userIndexes.length);
  }

  private double estimate(int userIndex, int itemIndex, int feature, double cachedEstimate, boolean trailing) {
    double sum = cachedEstimate;
    sum += userFeatures[userIndex][feature] * itemFeatures[itemIndex][feature];
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

  public double train(int userIndex, int itemIndex, int feature, double original, double cachedEstimate) {
    double error = original - estimate(userIndex, itemIndex, feature, cachedEstimate, true);
    double[] userVector = userFeatures[userIndex];
    double[] itemVector = itemFeatures[itemIndex];

    userVector[feature] += learningRate * (error * itemVector[feature] - preventOverfitting * userVector[feature]);
    itemVector[feature] += learningRate * (error * userVector[feature] - preventOverfitting * itemVector[feature]);

    return error;
  }

  protected void shufflePreferences() {
    /* Durstenfeld shuffle */
    for (int currentPos = userIndexes.length - 1; currentPos > 0; currentPos--) {
      int swapPos = random.nextInt(currentPos + 1);
      swapPreferences(currentPos, swapPos);
    }
  }

  private void swapPreferences(int posA, int posB) {
    int tmpUserIndex = userIndexes[posA];
    int tmpItemIndex = itemIndexes[posA];
    float tmpValue = values[posA];
    double tmpEstimate = cachedEstimates[posA];

    userIndexes[posA] = userIndexes[posB];
    itemIndexes[posA] = itemIndexes[posB];
    values[posA] = values[posB];
    cachedEstimates[posA] = cachedEstimates[posB];

    userIndexes[posB] = tmpUserIndex;
    itemIndexes[posB] = tmpItemIndex;
    values[posB] = tmpValue;
    cachedEstimates[posB] = tmpEstimate;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
