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

/** Calculates the SVD using an Expectation Maximization algorithm. */
public final class ExpectationMaximizationSVDFactorizer extends AbstractFactorizer {

  private static final Logger log = LoggerFactory.getLogger(ExpectationMaximizationSVDFactorizer.class);

  private final double learningRate;
  /** Parameter used to prevent overfitting. 0.02 is a good value. */
  private final double preventOverfitting;
  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** number of iterations */
  private final int numIterations;
  private final double randomNoise;
  /** user singular vectors */
  private double[][] leftVectors;
  /** item singular vectors */
  private double[][] rightVectors;
  private final DataModel dataModel;
  private List<SVDPreference> cachedPreferences;
  private double defaultValue;
  private double interval;

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
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.numIterations = numIterations;

    this.learningRate = learningRate;
    this.preventOverfitting = preventOverfitting;
    this.randomNoise = randomNoise;

  }

  @Override
  public Factorization factorize() throws TasteException {
    Random random = RandomUtils.getRandom();
    leftVectors = new double[dataModel.getNumUsers()][numFeatures];
    rightVectors = new double[dataModel.getNumItems()][numFeatures];

    double average = getAveragePreference();

    double prefInterval = dataModel.getMaxPreference() - dataModel.getMinPreference();
    defaultValue = Math.sqrt((average - prefInterval * 0.1) / numFeatures);
    interval = prefInterval * 0.1 / numFeatures;

    for (int feature = 0; feature < numFeatures; feature++) {
      for (int userIndex = 0; userIndex < dataModel.getNumUsers(); userIndex++) {
        leftVectors[userIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
      for (int itemIndex = 0; itemIndex < dataModel.getNumItems(); itemIndex++) {
        rightVectors[itemIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * randomNoise;
      }
    }
    cachedPreferences = Lists.newArrayListWithCapacity(dataModel.getNumUsers());
    cachePreferences();
    double rmse = dataModel.getMaxPreference() - dataModel.getMinPreference();
    for (int ii = 0; ii < numFeatures; ii++) {
      Collections.shuffle(cachedPreferences, random);
      for (int i = 0; i < numIterations; i++) {
        double err = 0.0;
        for (SVDPreference pref : cachedPreferences) {
          int useridx = userIndex(pref.getUserID());
          int itemidx = itemIndex(pref.getItemID());
          err += Math.pow(train(useridx, itemidx, ii, pref), 2.0);
        }
        rmse = Math.sqrt(err / cachedPreferences.size());
      }
      if (ii < numFeatures - 1) {
        for (SVDPreference pref : cachedPreferences) {
          int useridx = userIndex(pref.getUserID());
          int itemidx = itemIndex(pref.getItemID());
          buildCache(useridx, itemidx, ii, pref);
        }
      }
      log.info("Finished training feature {} with RMSE {}.", ii, rmse);
    }
    return createFactorization(leftVectors, rightVectors);
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

  private double train(int i, int j, int f, SVDPreference pref) {
    double[] leftVectorI = leftVectors[i];
    double[] rightVectorJ = rightVectors[j];
    double prediction = predictRating(i, j, f, pref, true);
    double err = pref.getValue() - prediction;
    leftVectorI[f] += learningRate * (err * rightVectorJ[f] - preventOverfitting * leftVectorI[f]);
    rightVectorJ[f] += learningRate * (err * leftVectorI[f] - preventOverfitting * rightVectorJ[f]);
    return err;
  }

  private void buildCache(int i, int j, int k, SVDPreference pref) {
    pref.setCache(predictRating(i, j, k, pref, false));
  }

  private double predictRating(int i, int j, int f, SVDPreference pref, boolean trailing) {
    float minPreference = dataModel.getMinPreference();
    float maxPreference = dataModel.getMaxPreference();
    double sum = pref.getCache();
    sum += leftVectors[i][f] * rightVectors[j][f];
    if (trailing) {
      sum += (numFeatures - f - 1) * (defaultValue + interval) * (defaultValue + interval);
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
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
        cachedPreferences.add(new SVDPreference(pref.getUserID(), pref.getItemID(), pref.getValue(), 0.0));
      }
    }
  }

}
