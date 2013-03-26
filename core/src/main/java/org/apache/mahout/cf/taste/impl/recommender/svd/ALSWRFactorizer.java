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
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.als.ImplicitFeedbackAlternatingLeastSquaresSolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * factorizes the rating matrix using "Alternating-Least-Squares with Weighted-λ-Regularization" as described in
 * <a href="http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf">
 * "Large-scale Collaborative Filtering for the Netflix Prize"</a>
 *
 *  also supports the implicit feedback variant of this approach as described in "Collaborative Filtering for Implicit
 *  Feedback Datasets" available at http://research.yahoo.com/pub/2433
 */
public class ALSWRFactorizer extends AbstractFactorizer {

  private final DataModel dataModel;

  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** parameter to control the regularization */
  private final double lambda;
  /** number of iterations */
  private final int numIterations;

  private final boolean usesImplicitFeedback;
  /** confidence weighting parameter, only necessary when working with implicit feedback */
  private final double alpha;

  private final int numTrainingThreads;

  private static final double DEFAULT_ALPHA = 40;

  private static final Logger log = LoggerFactory.getLogger(ALSWRFactorizer.class);

  public ALSWRFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      boolean usesImplicitFeedback, double alpha, int numTrainingThreads) throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.numIterations = numIterations;
    this.usesImplicitFeedback = usesImplicitFeedback;
    this.alpha = alpha;
    this.numTrainingThreads = numTrainingThreads;
  }

  public ALSWRFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
                         boolean usesImplicitFeedback, double alpha) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, usesImplicitFeedback, alpha,
        Runtime.getRuntime().availableProcessors());
  }

  public ALSWRFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, false, DEFAULT_ALPHA);
  }

  static class Features {

    private final DataModel dataModel;
    private final int numFeatures;

    private final double[][] M;
    private final double[][] U;

    Features(ALSWRFactorizer factorizer) throws TasteException {
      dataModel = factorizer.dataModel;
      numFeatures = factorizer.numFeatures;
      Random random = RandomUtils.getRandom();
      M = new double[dataModel.getNumItems()][numFeatures];
      LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
      while (itemIDsIterator.hasNext()) {
        long itemID = itemIDsIterator.nextLong();
        int itemIDIndex = factorizer.itemIndex(itemID);
        M[itemIDIndex][0] = averateRating(itemID);
        for (int feature = 1; feature < numFeatures; feature++) {
          M[itemIDIndex][feature] = random.nextDouble() * 0.1;
        }
      }
      U = new double[dataModel.getNumUsers()][numFeatures];
    }

    double[][] getM() {
      return M;
    }

    double[][] getU() {
      return U;
    }

    Vector getUserFeatureColumn(int index) {
      return new DenseVector(U[index]);
    }

    Vector getItemFeatureColumn(int index) {
      return new DenseVector(M[index]);
    }

    void setFeatureColumnInU(int idIndex, Vector vector) {
      setFeatureColumn(U, idIndex, vector);
    }

    void setFeatureColumnInM(int idIndex, Vector vector) {
      setFeatureColumn(M, idIndex, vector);
    }

    protected void setFeatureColumn(double[][] matrix, int idIndex, Vector vector) {
      for (int feature = 0; feature < numFeatures; feature++) {
        matrix[idIndex][feature] = vector.get(feature);
      }
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

  @Override
  public Factorization factorize() throws TasteException {
    log.info("starting to compute the factorization...");
    final Features features = new Features(this);

    /* feature maps necessary for solving for implicit feedback */
    OpenIntObjectHashMap<Vector> userY = null;
    OpenIntObjectHashMap<Vector> itemY = null;

    if (usesImplicitFeedback) {
      userY = userFeaturesMapping(dataModel.getUserIDs(), dataModel.getNumUsers(), features.getU());
      itemY = itemFeaturesMapping(dataModel.getItemIDs(), dataModel.getNumItems(), features.getM());
    }

    for (int iteration = 0; iteration < numIterations; iteration++) {
      log.info("iteration {}", iteration);

      /* fix M - compute U */
      ExecutorService queue = createQueue();
      LongPrimitiveIterator userIDsIterator = dataModel.getUserIDs();
      try {

        final ImplicitFeedbackAlternatingLeastSquaresSolver implicitFeedbackSolver = usesImplicitFeedback
            ? new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, itemY) : null;

        while (userIDsIterator.hasNext()) {
          final long userID = userIDsIterator.nextLong();
          final LongPrimitiveIterator itemIDsFromUser = dataModel.getItemIDsFromUser(userID).iterator();
          final PreferenceArray userPrefs = dataModel.getPreferencesFromUser(userID);
          queue.execute(new Runnable() {
            @Override
            public void run() {
              List<Vector> featureVectors = Lists.newArrayList();
              while (itemIDsFromUser.hasNext()) {
                long itemID = itemIDsFromUser.nextLong();
                featureVectors.add(features.getItemFeatureColumn(itemIndex(itemID)));
              }

              Vector userFeatures = usesImplicitFeedback
                  ? implicitFeedbackSolver.solve(sparseUserRatingVector(userPrefs))
                  : AlternatingLeastSquaresSolver.solve(featureVectors, ratingVector(userPrefs), lambda, numFeatures);

              features.setFeatureColumnInU(userIndex(userID), userFeatures);
            }
          });
        }
      } finally {
        queue.shutdown();
        try {
          queue.awaitTermination(dataModel.getNumUsers(), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          log.warn("Error when computing user features", e);
        }
      }

      /* fix U - compute M */
      queue = createQueue();
      LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
      try {

        final ImplicitFeedbackAlternatingLeastSquaresSolver implicitFeedbackSolver = usesImplicitFeedback
            ? new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, userY) : null;

        while (itemIDsIterator.hasNext()) {
          final long itemID = itemIDsIterator.nextLong();
          final PreferenceArray itemPrefs = dataModel.getPreferencesForItem(itemID);
          queue.execute(new Runnable() {
            @Override
            public void run() {
              List<Vector> featureVectors = Lists.newArrayList();
              for (Preference pref : itemPrefs) {
                long userID = pref.getUserID();
                featureVectors.add(features.getUserFeatureColumn(userIndex(userID)));
              }

              Vector itemFeatures = usesImplicitFeedback
                  ? implicitFeedbackSolver.solve(sparseItemRatingVector(itemPrefs))
                  : AlternatingLeastSquaresSolver.solve(featureVectors, ratingVector(itemPrefs), lambda, numFeatures);

              features.setFeatureColumnInM(itemIndex(itemID), itemFeatures);
            }
          });
        }
      } finally {
        queue.shutdown();
        try {
          queue.awaitTermination(dataModel.getNumItems(), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          log.warn("Error when computing item features", e);
        }
      }
    }

    log.info("finished computation of the factorization...");
    return createFactorization(features.getU(), features.getM());
  }

  protected ExecutorService createQueue() {
    return Executors.newFixedThreadPool(numTrainingThreads);
  }

  protected static Vector ratingVector(PreferenceArray prefs) {
    double[] ratings = new double[prefs.length()];
    for (int n = 0; n < prefs.length(); n++) {
      ratings[n] = prefs.get(n).getValue();
    }
    return new DenseVector(ratings, true);
  }

  //TODO find a way to get rid of the object overhead here
  protected OpenIntObjectHashMap<Vector> itemFeaturesMapping(LongPrimitiveIterator itemIDs, int numItems,
      double[][] featureMatrix) {
    OpenIntObjectHashMap<Vector> mapping = new OpenIntObjectHashMap<Vector>(numItems);
    while (itemIDs.hasNext()) {
      long itemID = itemIDs.next();
      mapping.put((int) itemID, new DenseVector(featureMatrix[itemIndex(itemID)], true));
    }

    return mapping;
  }

  protected OpenIntObjectHashMap<Vector> userFeaturesMapping(LongPrimitiveIterator userIDs, int numUsers,
      double[][] featureMatrix) {
    OpenIntObjectHashMap<Vector> mapping = new OpenIntObjectHashMap<Vector>(numUsers);

    while (userIDs.hasNext()) {
      long userID = userIDs.next();
      mapping.put((int) userID, new DenseVector(featureMatrix[userIndex(userID)], true));
    }

    return mapping;
  }

  protected Vector sparseItemRatingVector(PreferenceArray prefs) {
    SequentialAccessSparseVector ratings = new SequentialAccessSparseVector(Integer.MAX_VALUE, prefs.length());
    for (Preference preference : prefs) {
      ratings.set((int) preference.getUserID(), preference.getValue());
    }
    return ratings;
  }

  protected Vector sparseUserRatingVector(PreferenceArray prefs) {
    SequentialAccessSparseVector ratings = new SequentialAccessSparseVector(Integer.MAX_VALUE, prefs.length());
    for (Preference preference : prefs) {
      ratings.set((int) preference.getItemID(), preference.getValue());
    }
    return ratings;
  }
}
