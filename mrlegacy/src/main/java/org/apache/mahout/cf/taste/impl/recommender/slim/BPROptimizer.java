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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Bayesian Personalized Ranking Optimizer from Implicit Feedback.
 * http://www.ismll
 * .uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking
 * .pdf
 *
 */
public class BPROptimizer extends AbstractOptimizer {

  protected int nrIterations;

  protected double learnRate;

  protected double regPos;

  protected double regNeg;

  private int nrPosEvents;

  private static final Logger log = LoggerFactory.getLogger(BPROptimizer.class);

  public static final double DEFAULT_LEARN_RATE = 0.05;
  public static final double DEFAULT_REG_POS = 0.0025d;
  public static final double DEFAULT_REG_NEG = 0.00025d;
  public static final double DEFAULT_MEAN = 0.0d;
  public static final double DEFAULT_STDEV = 0.1d;

  public BPROptimizer(DataModel dataModel, int numIterations)
      throws TasteException {
    this(dataModel, numIterations, DEFAULT_LEARN_RATE, DEFAULT_REG_POS,
        DEFAULT_REG_NEG, DEFAULT_MEAN, DEFAULT_STDEV);
  }

  public BPROptimizer(DataModel dataModel, int numIterations, double learnRate)
      throws TasteException {
    this(dataModel, numIterations, learnRate, DEFAULT_REG_POS, DEFAULT_REG_NEG,
        DEFAULT_MEAN, DEFAULT_STDEV);
  }

  public BPROptimizer(DataModel dataModel, int numIterations, double learnRate,
      double regPos, double regNeg) throws TasteException {
    this(dataModel, numIterations, learnRate, regPos, regNeg, DEFAULT_MEAN,
        DEFAULT_STDEV);
  }

  public BPROptimizer(DataModel dataModel, int numIterations, double learnRate,
      double regPos, double regNeg, double mean, double stDev)
      throws TasteException {
    super(dataModel, mean, stDev);
    this.nrIterations = numIterations;
    this.learnRate = learnRate;
    this.regPos = regPos;
    this.regNeg = regNeg;
  }

  @Override
  protected void prepareTraining() throws TasteException {
    super.prepareTraining();

    // Initialize number of recorded events
    nrPosEvents = 0;
    LongPrimitiveIterator it = dataModel.getItemIDs();
    while (it.hasNext()) {
      long itemID = it.next();
      nrPosEvents += dataModel.getPreferencesForItem(itemID).length();
    }
  }

  @Override
  public SlimSolution findSolution() throws TasteException {
    prepareTraining();

    for (int i = 0; i <= nrIterations; i++) {
      if (i % 10 == 0) {
        log.info("iteration: " + i);
      }

      iterate();
    }

    return slim;
  }

  /*
   * One iteration of stochastic gradient ascent over the training data.
   */
  protected void iterate()
      throws TasteException {

    // uniform user sampling, without replacement
    for (int i = 0; i < nrPosEvents; i++) {
      long userID = sampleUserID();
      PreferenceArray userItems = dataModel.getPreferencesFromUser(userID);

      int itemPosIndex = samplePosItemIndex(userItems);
      int itemNegIndex = sampleNegItemIndex(userItems);
      updateParameters(userID, userItems, itemPosIndex, itemNegIndex);
    }
  }

  protected void updateParameters(long userID, PreferenceArray userItems,
      int itemPosIndex, int itemNegIndex) {

    double diff = predictWithDifference(userID, itemPosIndex, itemNegIndex);
    double sigmoid = 1.0d / (1.0d + Math.exp(diff));

    Matrix itemWeights = slim.getItemWeights();

    for (int i = 0; i < userItems.length(); i++) {
      long itemID = userItems.getItemID(i);
      int itemIndex = itemIndex(itemID);
      double weightPosItem = getAndInitWeight(itemWeights, itemPosIndex,
          itemIndex);
      double weightNegItem = getAndInitWeight(itemWeights, itemNegIndex,
          itemIndex);

      if (itemIndex != itemPosIndex) {
        double update = sigmoid - regPos * weightPosItem;
        update = (weightPosItem + learnRate * update);
        itemWeights.setQuick(itemPosIndex, itemIndex, update);
      }

      if (itemIndex != itemNegIndex) {
        double update = -sigmoid - regNeg * weightNegItem;
        update = (weightNegItem + learnRate * update);
        itemWeights.setQuick(itemNegIndex, itemIndex, update);
      }
    }
  }

  protected double predictWithDifference(long userID, int itemPosIndex,
      int itemNegIndex) {
    try {
      PreferenceArray userItems = dataModel.getPreferencesFromUser(userID);

      Matrix itemWeights = slim.getItemWeights();
      double prediction = 0;

      for (int i = 0; i < userItems.length(); i++) {
        long itemID = userItems.getItemID(i);
        int itemIndex = itemIndex(itemID);
        prediction += getAndInitWeight(itemWeights, itemPosIndex, itemIndex)
            - getAndInitWeight(itemWeights, itemNegIndex, itemIndex);
      }

      return prediction;

    } catch (TasteException e) {
      return Double.MIN_VALUE;
    }
  }

  /**
   * Number of iterations to perform.
   */
  public int getNrIterations() {
    return nrIterations;
  }

  /**
   * Get learning rate.
   */
  public double getLearningRate() {
    return learnRate;
  }

  /**
   * Get regularization parameter for positive items.
   */
  public double getRegPositive() {
    return regPos;
  }

  /**
   * Get regularization parameter for negative items.
   */
  public double getRegNegative() {
    return regNeg;
  }

  public class BPRExplanation {

    private final int iteration;
    private final int maxIter;
    private final SlimSolution solution;
    private final double updateAvg;
    private final double aucAvg;

    public BPRExplanation(final SlimSolution solution, final int iteration,
        final int maxIter, final double updateAvg, final double aucAvg) {
      this.iteration = iteration;
      this.maxIter = maxIter;
      this.solution = solution;
      this.updateAvg = updateAvg;
      this.aucAvg = aucAvg;
    }

    public double getUpdateAvg() {
      return updateAvg;
    }

    public double getAucAvg() {
      return aucAvg;
    }

    public int getIteration() {
      return iteration;
    }

    public int getMaxIteration() {
      return maxIter;
    }

    public SlimSolution getSolution() {
      return solution;
    }
  }

}
