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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.common.RandomUtils;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * SVD++, an enhancement of classical matrix factorization for rating prediction.
 * Additionally to using ratings (how did people rate?) for learning, this model also takes into account
 * who rated what.
 *
 * Yehuda Koren: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model, KDD 2008.
 * http://research.yahoo.com/files/kdd08koren.pdf
 */
public final class SVDPlusPlusFactorizer extends RatingSGDFactorizer {

  private double[][] p;
  private double[][] y;
  private Map<Integer, List<Integer>> itemsByUser;

  public SVDPlusPlusFactorizer(DataModel dataModel, int numFeatures, int numIterations) throws TasteException {
    this(dataModel, numFeatures, 0.01, 0.1, 0.01, numIterations, 1.0);
    biasLearningRate = 0.7;
    biasReg = 0.33;
  }

  public SVDPlusPlusFactorizer(DataModel dataModel, int numFeatures, double learningRate, double preventOverfitting,
      double randomNoise, int numIterations, double learningRateDecay) throws TasteException {
    super(dataModel, numFeatures, learningRate, preventOverfitting, randomNoise, numIterations, learningRateDecay);
  }

  @Override
  protected void prepareTraining() throws TasteException {
    super.prepareTraining();
    Random random = RandomUtils.getRandom();

    p = new double[dataModel.getNumUsers()][numFeatures];
    for (int i = 0; i < p.length; i++) {
      for (int feature = 0; feature < FEATURE_OFFSET; feature++) {
        p[i][feature] = 0;
      }
      for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
        p[i][feature] = random.nextGaussian() * randomNoise;
      }
    }

    y = new double[dataModel.getNumItems()][numFeatures];
    for (int i = 0; i < y.length; i++) {
      for (int feature = 0; feature < FEATURE_OFFSET; feature++) {
        y[i][feature] = 0;
      }
      for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
        y[i][feature] = random.nextGaussian() * randomNoise;
      }
    }

    /* get internal item IDs which we will need several times */
    itemsByUser = Maps.newHashMap();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      long userId = userIDs.nextLong();
      int userIndex = userIndex(userId);
      FastIDSet itemIDsFromUser = dataModel.getItemIDsFromUser(userId);
      List<Integer> itemIndexes = Lists.newArrayListWithCapacity(itemIDsFromUser.size());
      itemsByUser.put(userIndex, itemIndexes);
      for (long itemID2 : itemIDsFromUser) {
        int i2 = itemIndex(itemID2);
        itemIndexes.add(i2);
      }
    }
  }

  @Override
  public Factorization factorize() throws TasteException {
    prepareTraining();

    super.factorize();

    for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
      for (int itemIndex : itemsByUser.get(userIndex)) {
        for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
          userVectors[userIndex][feature] += y[itemIndex][feature];
        }
      }
      double denominator = Math.sqrt(itemsByUser.get(userIndex).size());
      for (int feature = 0; feature < userVectors[userIndex].length; feature++) {
        userVectors[userIndex][feature] =
            (float) (userVectors[userIndex][feature] / denominator + p[userIndex][feature]);
      }
    }

    return createFactorization(userVectors, itemVectors);
  }


  @Override
  protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
    int userIndex = userIndex(userID);
    int itemIndex = itemIndex(itemID);

    double[] userVector = p[userIndex];
    double[] itemVector = itemVectors[itemIndex];

    double[] pPlusY = new double[numFeatures];
    for (int i2 : itemsByUser.get(userIndex)) {
      for (int f = FEATURE_OFFSET; f < numFeatures; f++) {
        pPlusY[f] += y[i2][f];
      }
    }
    double denominator = Math.sqrt(itemsByUser.get(userIndex).size());
    for (int feature = 0; feature < pPlusY.length; feature++) {
      pPlusY[feature] = (float) (pPlusY[feature] / denominator + p[userIndex][feature]);
    }

    double prediction = predictRating(pPlusY, itemIndex);
    double err = rating - prediction;
    double normalized_error = err / denominator;

    // adjust user bias
    userVector[USER_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);

    // adjust item bias
    itemVector[ITEM_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);

    // adjust features
    for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
      double pF = userVector[feature];
      double iF = itemVector[feature];

      double deltaU = err * iF - preventOverfitting * pF;
      userVector[feature] += currentLearningRate * deltaU;

      double deltaI = err * pPlusY[feature] - preventOverfitting * iF;
      itemVector[feature] += currentLearningRate * deltaI;

      double commonUpdate = normalized_error * iF;
      for (int itemIndex2 : itemsByUser.get(userIndex)) {
        double deltaI2 = commonUpdate - preventOverfitting * y[itemIndex2][feature];
        y[itemIndex2][feature] += learningRate * deltaI2;
      }
    }
  }

  private double predictRating(double[] userVector, int itemID) {
    double sum = 0;
    for (int feature = 0; feature < numFeatures; feature++) {
      sum += userVector[feature] * itemVectors[itemID][feature];
    }
    return sum;
  }
}
