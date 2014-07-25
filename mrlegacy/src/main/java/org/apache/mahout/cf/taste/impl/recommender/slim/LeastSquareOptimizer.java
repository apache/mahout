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

package org.apache.mahout.cf.taste.impl.recommender.slim;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SLIM for item prediction (ranking) optimized for the elastic net loss. The
 * model is learned using a coordinate descent algorithm with soft thresholding.
 * http://www.jstatsoft.org/v33/i01/paper
 * 
 */
public class LeastSquareOptimizer extends AbstractOptimizer {

  /**
   * Number of iterations to perform.
   */
  protected int nrIterations;

  /**
   * Regularization parameter for the L1 regularization term (lambda).
   */
  protected double regLambda;

  /**
   * Regularization parameter for the L2 regularization term (beta/2).
   */
  protected double regBeta;

  /**
   * ItemBasedRecommender used in learning.
   */
  protected ItemBasedRecommender itemRecommender;

  /**
   * Number of items needed when getting most similar items.
   */
  protected int howManyItems;

  /**
   * Number of threads to run optimization.
   */
  protected int numThreads;

  private static final Logger log = LoggerFactory
      .getLogger(LeastSquareOptimizer.class);

  public static final int DEFAULT_NUM_ITERATIONS = 20;
  public static final int DEFAULT_NUM_ITEMS = 0;
  public static final int DEFAULT_NUM_THREADS = 1;
  public static final double DEFAULT_REG_LAMBDA = 0.01d;
  public static final double DEFAULT_REG_BETA = 0.001d;
  public static final double DEFAULT_REG_MEAN = 0.0d;
  public static final double DEFAULT_REG_STDEV = 0.1d;

  public LeastSquareOptimizer(DataModel dataModel) throws TasteException {
    this(dataModel, DEFAULT_NUM_THREADS, DEFAULT_NUM_ITERATIONS,
        DEFAULT_NUM_ITEMS, DEFAULT_REG_LAMBDA, DEFAULT_REG_BETA,
        DEFAULT_REG_MEAN, DEFAULT_REG_STDEV);
  }

  public LeastSquareOptimizer(DataModel dataModel, int numThreads,
      int nrIterations) throws TasteException {
    this(dataModel, numThreads, nrIterations, DEFAULT_NUM_ITEMS,
        DEFAULT_REG_LAMBDA, DEFAULT_REG_BETA, DEFAULT_REG_MEAN,
        DEFAULT_REG_STDEV);
  }

  public LeastSquareOptimizer(DataModel dataModel, int numThreads,
      int nrIterations, int howManyItems) throws TasteException {
    this(dataModel, numThreads, nrIterations, howManyItems, DEFAULT_REG_LAMBDA,
        DEFAULT_REG_BETA, DEFAULT_REG_MEAN, DEFAULT_REG_STDEV);
  }

  public LeastSquareOptimizer(DataModel dataModel, int numThreads,
      int nrIterations, int howManyItems, double regLambda, double regBeta)
      throws TasteException {
    this(dataModel, numThreads, nrIterations, howManyItems, regLambda, regBeta,
        DEFAULT_REG_MEAN, DEFAULT_REG_STDEV);
  }

  public LeastSquareOptimizer(DataModel dataModel, int numThreads,
      int nrIterations, int howManyItems, double regLambda, double regBeta,
      double mean, double stDev) throws TasteException {
    super(dataModel, mean, stDev);
    this.nrIterations = nrIterations;
    this.howManyItems = howManyItems;
    this.regLambda = regLambda;
    this.regBeta = regBeta;
    this.numThreads = numThreads;
  }

  @Override
  protected void prepareTraining() throws TasteException {
    super.prepareTraining();

    if (howManyItems > 0) {
      ItemSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
      itemRecommender = new GenericItemBasedRecommender(dataModel, similarity);
    }
  }

  @Override
  public SlimSolution findSolution() throws TasteException {
    prepareTraining();
    LongPrimitiveIterator it = dataModel.getItemIDs();

    // learning for each item can be parallelized
    ExecutorService executor = Executors.newFixedThreadPool(numThreads);

    while (it.hasNext()) {
      final long itemID = it.next();

      executor.execute(new Runnable() {

        @Override
        public void run() {
          try {
            for (int i = 0; i <= nrIterations; i++) {
              iterate(itemID);
            }

            log.info("Learned item: " + itemID);

          } catch (TasteException ex) {
            log.error(ex.toString());
          }
        }
      });
    }

    try {
      executor.shutdown();
      executor.awaitTermination(1, TimeUnit.DAYS);
    } catch (InterruptedException e) {
      throw new TasteException("waiting fof termination interrupted", e);
    }

    return slim;
  }

  protected void iterate(final long itemID) throws TasteException {

    if (howManyItems > 0) {
      List<RecommendedItem> similarItems = itemRecommender.mostSimilarItems(
          itemID, howManyItems);
      if (similarItems.size() >= howManyItems) {
        for (RecommendedItem similarItem : similarItems) {
          long similarItemID = similarItem.getItemID();
          if (similarItemID != itemID) {
            updateParameters(itemID, similarItemID);

          }
        }
        return;
      }
    }

    LongPrimitiveIterator it = dataModel.getItemIDs();

    while (it.hasNext()) {
      long item2ID = it.next();
      if (itemID != item2ID) {
        updateParameters(itemID, item2ID);

      }
    }
  }

  /**
   * Update item parameters according to the coordinate descent update rule.
   * 
   * @param itemID
   *          An item ID
   * @param similarItemID
   *          An item similar to itemID
   * @throws TasteException
   */
  protected void updateParameters(final long itemID, final long similarItemID)
      throws TasteException {
    PreferenceArray userPrefs = dataModel.getPreferencesForItem(similarItemID);

    double gradient = 0;
    for (Preference pref : userPrefs) {
      long userID = pref.getUserID();

      if (dataModel.getPreferenceValue(userID, itemID) != null) {
        gradient += 1;
      }

      gradient -= predictWithExclusion(userID, itemID, similarItemID);
    }

    gradient /= (dataModel.getNumUsers() + 1);

    int itemIndex = itemIndex(itemID);
    int similarItemIndex = itemIndex(similarItemID);
    Matrix itemWeights = slim.getItemWeights();

    if (regLambda < Math.abs(gradient)) {
      double update;
      if (gradient > 0) {
        update = (gradient - regLambda) / (1.0 + regBeta);
        itemWeights.setQuick(itemIndex, similarItemIndex, update);
      } else {
        update = (gradient + regLambda) / (1.0 + regBeta);
        itemWeights.setQuick(itemIndex, similarItemIndex, update);
      }
    } else {
      // the default value is 0
      // itemWeights.setQuick(itemIndex, similarItemIndex, 0.0);
    }
  }

  protected double predictWithExclusion(final long userID, final long itemID,
      final long excludeItemID) {
    try {
      Matrix itemWeights = slim.getItemWeights();
      double prediction = 0;
      int itemIndex = itemIndex(itemID);

      if (howManyItems > 0) {
        List<RecommendedItem> similarItems = itemRecommender.mostSimilarItems(
            itemID, howManyItems);
        if (similarItems.size() >= howManyItems) {
          for (RecommendedItem similarItem : similarItems) {
            long similarID = similarItem.getItemID();
            Float pref = dataModel.getPreferenceValue(userID, similarID);
            if (pref != null && similarID != excludeItemID) {
              int item2Index = itemIndex(similarID);
              prediction += getAndInitWeightPos(itemWeights, itemIndex,
                  item2Index);
            }
          }
          return prediction;
        }
      }

      PreferenceArray userItems = dataModel.getPreferencesFromUser(userID);
      Iterator<Preference> it = userItems.iterator();
      while (it.hasNext()) {
        Preference pref = it.next();
        long prefItemID = pref.getItemID();
        if (prefItemID != excludeItemID) {
          int item2Index = itemIndex(prefItemID);
          prediction += getAndInitWeightPos(itemWeights, itemIndex, item2Index);
        }
      }

      return prediction;

    } catch (TasteException e) {
      return 0;
    }
  }

  /**
   * Number of iterations to perform.
   */
  public int getNrIterations() {
    return nrIterations;
  }

  /**
   * Number of items needed when getting most similar items.
   */
  public int getHowManyItems() {
    return howManyItems;
  }

  /**
   * Regularization parameter for the L1 regularization term (lambda).
   */
  public double getRegLambda() {
    return regLambda;
  }

  /**
   * Regularization parameter for the L2 regularization term (beta/2).
   */
  public double getRegBeta() {
    return regBeta;
  }

}
