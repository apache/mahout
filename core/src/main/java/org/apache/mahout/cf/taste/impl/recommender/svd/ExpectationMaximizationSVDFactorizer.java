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
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Uses Single Value Decomposition to find the main features of the data set. Thanks to Simon Funk for the hints
 * in the implementation, {@see http://sifter.org/~simon/journal/20061211.html}.
 */
public class ExpectationMaximizationSVDFactorizer extends AbstractFactorizer {

  private final Random random;

  private final double learningRate;
  /** Parameter used to prevent overfitting. 0.02 is a good value. */
  private final double preventOverfitting;

  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** number of iterations */
  private final int numIterations;

  /** user singular vectors */
  private final double[][] leftVectors;

  /** item singular vectors */
  private final double[][] rightVectors;

  private final DataModel dataModel;
  private final List<Preference> cachedPreferences;

  private static final Logger log = LoggerFactory.getLogger(ExpectationMaximizationSVDFactorizer.class);

  public ExpectationMaximizationSVDFactorizer(DataModel dataModel,
                                              int numFeatures,
                                              int numIterations) throws TasteException {
    // use the default parameters from the old SVDRecommender implementation    
    this(dataModel, numFeatures, 0.005, 0.02, 0.005, numIterations);
  }

  public ExpectationMaximizationSVDFactorizer(DataModel dataModel,
                                              int numFeatures,
                                              double learningRate,
                                              double preventOverfitting,
                                              double randomNoise,
                                              int numIterations) throws TasteException {
    super(dataModel);
    random = RandomUtils.getRandom();
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.numIterations = numIterations;

    this.learningRate = learningRate;
    this.preventOverfitting = preventOverfitting;

    leftVectors = new double[dataModel.getNumUsers()][numFeatures];
    rightVectors = new double[dataModel.getNumItems()][numFeatures];

    double average = getAveragePreference();
    double defaultValue = Math.sqrt((average - 1.0) / numFeatures);

    for (int feature = 0; feature < numFeatures; feature++) {
      for (int userIndex = 0; userIndex < dataModel.getNumUsers(); userIndex++) {
        leftVectors[userIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * randomNoise;
      }
      for (int itemIndex = 0; itemIndex < dataModel.getNumItems(); itemIndex++) {
        rightVectors[itemIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * randomNoise;
      }
    }

    cachedPreferences = new ArrayList<Preference>(dataModel.getNumUsers());
  }

  @Override
  public Factorization factorize() throws TasteException {
    log.info("starting to compute the factorization...");

    cachePreferences();
    for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
      log.info("iteration {}", currentIteration);
      nextTrainStep();
    }

    log.info("finished computation of the factorization...");
    return createFactorization(leftVectors, rightVectors);
  }

  void cachePreferences() throws TasteException {
    cachedPreferences.clear();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
        cachedPreferences.add(pref);
      }
    }
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

  void nextTrainStep() {
    Collections.shuffle(cachedPreferences, random);
    for (int feature = 0; feature < numFeatures; feature++) {
      for (Preference pref : cachedPreferences) {
        train(userIndex(pref.getUserID()), itemIndex(pref.getItemID()), feature, pref.getValue());
      }
    }
  }

  double getDotProduct(int userIndex, int itemIndex) {
    double result = 1.0;
    for (int feature = 0; feature < this.numFeatures; feature++) {
      result += leftVectors[userIndex][feature] * rightVectors[itemIndex][feature];
    }
    return result;
  }

  void train(int userIndex, int itemIndex, int currentFeature, double value) {
    double err = value - getDotProduct(userIndex, itemIndex);
    double[] leftVector = leftVectors[userIndex];
    double[] rightVector = rightVectors[itemIndex];
    leftVector[currentFeature] +=
        learningRate * (err * rightVector[currentFeature] - preventOverfitting * leftVector[currentFeature]);
    rightVector[currentFeature] +=
        learningRate * (err * leftVector[currentFeature] - preventOverfitting * rightVector[currentFeature]);
  }

}
