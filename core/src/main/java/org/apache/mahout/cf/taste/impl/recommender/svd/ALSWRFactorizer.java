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
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternateLeastSquaresSolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * factorizes the rating matrix using "Alternating-Least-Squares with Weighted-λ-Regularization" as described in
 * the paper "Large-scale Collaborative Filtering for the Netflix Prize" available at
 * {@see http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf}
 */
public class ALSWRFactorizer extends AbstractFactorizer {

  private final DataModel dataModel;

  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** parameter to control the regularization */
  private final double lambda;
  /** number of iterations */
  private final int numIterations;

  private static final Logger log = LoggerFactory.getLogger(ALSWRFactorizer.class);

  public ALSWRFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations) throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.numIterations = numIterations;
  }

  @Override
  public Factorization factorize() throws TasteException {
    log.info("starting to compute the factorization...");
    AlternateLeastSquaresSolver solver = new AlternateLeastSquaresSolver();

    double[][] M = initializeM();
    double[][] U = null;

    for (int iteration = 0; iteration < numIterations; iteration++) {
      log.info("iteration {}", iteration);

      /* fix M - compute U */
      U = new double[dataModel.getNumUsers()][numFeatures];

      LongPrimitiveIterator userIDsIterator = dataModel.getUserIDs();
      while (userIDsIterator.hasNext()) {
        long userID = userIDsIterator.nextLong();
        List<Vector> featureVectors = new ArrayList<Vector>();
        LongPrimitiveIterator itemIDsFromUser = dataModel.getItemIDsFromUser(userID).iterator();
        while (itemIDsFromUser.hasNext()) {
          long itemID = itemIDsFromUser.nextLong();
          featureVectors.add(new DenseVector(M[itemIndex(itemID)]));
        }
        PreferenceArray userPrefs = dataModel.getPreferencesFromUser(userID);
        Vector userFeatures = solver.solve(featureVectors, ratingVector(userPrefs), lambda, numFeatures);
        setFeatureColumn(U, userIndex(userID), userFeatures);
      }

      /* fix U - compute M */
      M = new double[dataModel.getNumItems()][numFeatures];

      LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
      while (itemIDsIterator.hasNext()) {
        long itemID = itemIDsIterator.nextLong();
        List<Vector> featureVectors = new ArrayList<Vector>();
        for (Preference pref : dataModel.getPreferencesForItem(itemID)) {
          long userID = pref.getUserID();
          featureVectors.add(new DenseVector(U[userIndex(userID)]));
        }
        PreferenceArray itemPrefs = dataModel.getPreferencesForItem(itemID);
        Vector itemFeatures = solver.solve(featureVectors, ratingVector(itemPrefs), lambda, numFeatures);
        setFeatureColumn(M, itemIndex(itemID), itemFeatures);
      }
    }

    log.info("finished computation of the factorization...");
    return createFactorization(U, M);
  }

  protected double[][] initializeM() throws TasteException {
    Random random = RandomUtils.getRandom();
    double[][] M = new double[dataModel.getNumItems()][numFeatures];

    LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
    while (itemIDsIterator.hasNext()) {
      long itemID = itemIDsIterator.nextLong();
      int itemIDIndex = itemIndex(itemID);
      M[itemIDIndex][0] = averateRating(itemID);
      for (int feature = 1; feature < numFeatures; feature++) {
        M[itemIDIndex][feature] = random.nextDouble() * 0.1;
      }
    }
    return M;
  }

  protected void setFeatureColumn(double[][] matrix, int idIndex, Vector vector) {
    for (int feature = 0; feature < numFeatures; feature++) {
      matrix[idIndex][feature] = vector.get(feature);
    }
  }

  protected Vector ratingVector(PreferenceArray prefs) {
    double[] ratings = new double[prefs.length()];
    for (int n = 0; n < prefs.length(); n++) {
      ratings[n] = prefs.get(n).getValue();
    }
    return new DenseVector(ratings);
  }

  protected double averateRating(long itemID) throws TasteException {
    PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
    RunningAverage avg = new FullRunningAverage();
    for (Preference pref : prefs) {
      avg.addDatum(pref.getValue());
    }
    return avg.getAverage();
  }

}
