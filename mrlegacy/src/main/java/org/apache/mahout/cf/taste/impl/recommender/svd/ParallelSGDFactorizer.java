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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Minimalistic implementation of Parallel SGD factorizer based on
 * <a href="http://www.sze.hu/~gtakacs/download/jmlr_2009.pdf">
 * "Scalable Collaborative Filtering Approaches for Large Recommender Systems"</a>
 * and
 * <a href="hwww.cs.wisc.edu/~brecht/papers/hogwildTR.pdf">
 * "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"</a> */
public class ParallelSGDFactorizer extends AbstractFactorizer {

  private final DataModel dataModel;
  /** Parameter used to prevent overfitting. */
  private final double lambda;
  /** Number of features used to compute this factorization */
  private final int rank;
  /** Number of iterations */
  private final int numEpochs;

  private int numThreads;

  // these next two control decayFactor^steps exponential type of annealing learning rate and decay factor
  private double mu0 = 0.01;
  private double decayFactor = 1;
  // these next two control 1/steps^forget type annealing
  private int stepOffset = 0;
  // -1 equals even weighting of all examples, 0 means only use exponential annealing
  private double forgettingExponent = 0;

  // The following two should be inversely proportional :)
  private double biasMuRatio = 0.5;
  private double biasLambdaRatio = 0.1;

  /** TODO: this is not safe as += is not atomic on many processors, can be replaced with AtomicDoubleArray
   * but it works just fine right now  */
  /** user features */
  protected volatile double[][] userVectors;
  /** item features */
  protected volatile double[][] itemVectors;

  private final PreferenceShuffler shuffler;

  private int epoch = 1;
  /** place in user vector where the bias is stored */
  private static final int USER_BIAS_INDEX = 1;
  /** place in item vector where the bias is stored */
  private static final int ITEM_BIAS_INDEX = 2;
  private static final int FEATURE_OFFSET = 3;
  /** Standard deviation for random initialization of features */
  private static final double NOISE = 0.02;

  private static final Logger logger = LoggerFactory.getLogger(ParallelSGDFactorizer.class);

  protected static class PreferenceShuffler {

    private Preference[] preferences;
    private Preference[] unstagedPreferences;

    protected final RandomWrapper random = RandomUtils.getRandom();

    public PreferenceShuffler(DataModel dataModel) throws TasteException {
      cachePreferences(dataModel);
      shuffle();
      stage();
    }

    private int countPreferences(DataModel dataModel) throws TasteException {
      int numPreferences = 0;
      LongPrimitiveIterator userIDs = dataModel.getUserIDs();
      while (userIDs.hasNext()) {
        PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userIDs.nextLong());
        numPreferences += preferencesFromUser.length();
      }
      return numPreferences;
    }

    private void cachePreferences(DataModel dataModel) throws TasteException {
      int numPreferences = countPreferences(dataModel);
      preferences = new Preference[numPreferences];

      LongPrimitiveIterator userIDs = dataModel.getUserIDs();
      int index = 0;
      while (userIDs.hasNext()) {
        long userID = userIDs.nextLong();
        PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userID);
        for (Preference preference : preferencesFromUser) {
          preferences[index++] = preference;
        }
      }
    }

    public void shuffle() {
      unstagedPreferences = preferences.clone();
      /* Durstenfeld shuffle */
      for (int i = unstagedPreferences.length - 1; i > 0; i--) {
        int rand = random.nextInt(i + 1);
        swapCachedPreferences(i, rand);
      }
    }

    //merge this part into shuffle() will make compiler-optimizer do some real absurd stuff, test on OpenJDK7
    private void swapCachedPreferences(int x, int y) {
      Preference p = unstagedPreferences[x];

      unstagedPreferences[x] = unstagedPreferences[y];
      unstagedPreferences[y] = p;
    }

    public void stage() {
      preferences = unstagedPreferences;
    }

    public Preference get(int i) {
      return preferences[i];
    }

    public int size() {
      return preferences.length;
    }

  }

  public ParallelSGDFactorizer(DataModel dataModel, int numFeatures, double lambda, int numEpochs)
    throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.rank = numFeatures + FEATURE_OFFSET;
    this.lambda = lambda;
    this.numEpochs = numEpochs;

    shuffler = new PreferenceShuffler(dataModel);

    //max thread num set to n^0.25 as suggested by hogwild! paper
    numThreads = Math.min(Runtime.getRuntime().availableProcessors(), (int) Math.pow((double) shuffler.size(), 0.25));
  }

  public ParallelSGDFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      double mu0, double decayFactor, int stepOffset, double forgettingExponent) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations);

    this.mu0 = mu0;
    this.decayFactor = decayFactor;
    this.stepOffset = stepOffset;
    this.forgettingExponent = forgettingExponent;
  }

  public ParallelSGDFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      double mu0, double decayFactor, int stepOffset, double forgettingExponent, int numThreads) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, mu0, decayFactor, stepOffset, forgettingExponent);

    this.numThreads = numThreads;
  }

  public ParallelSGDFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      double mu0, double decayFactor, int stepOffset, double forgettingExponent,
      double biasMuRatio, double biasLambdaRatio) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, mu0, decayFactor, stepOffset, forgettingExponent);

    this.biasMuRatio = biasMuRatio;
    this.biasLambdaRatio = biasLambdaRatio;
  }

  public ParallelSGDFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      double mu0, double decayFactor, int stepOffset, double forgettingExponent,
      double biasMuRatio, double biasLambdaRatio, int numThreads) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, mu0, decayFactor, stepOffset, forgettingExponent, biasMuRatio,
         biasLambdaRatio);

    this.numThreads = numThreads;
  }

  protected void initialize() throws TasteException {
    RandomWrapper random = RandomUtils.getRandom();
    userVectors = new double[dataModel.getNumUsers()][rank];
    itemVectors = new double[dataModel.getNumItems()][rank];

    double globalAverage = getAveragePreference();
    for (int userIndex = 0; userIndex < userVectors.length; userIndex++) {
      userVectors[userIndex][0] = globalAverage;
      userVectors[userIndex][USER_BIAS_INDEX] = 0; // will store user bias
      userVectors[userIndex][ITEM_BIAS_INDEX] = 1; // corresponding item feature contains item bias
      for (int feature = FEATURE_OFFSET; feature < rank; feature++) {
        userVectors[userIndex][feature] = random.nextGaussian() * NOISE;
      }
    }
    for (int itemIndex = 0; itemIndex < itemVectors.length; itemIndex++) {
      itemVectors[itemIndex][0] = 1; // corresponding user feature contains global average
      itemVectors[itemIndex][USER_BIAS_INDEX] = 1; // corresponding user feature contains user bias
      itemVectors[itemIndex][ITEM_BIAS_INDEX] = 0; // will store item bias
      for (int feature = FEATURE_OFFSET; feature < rank; feature++) {
        itemVectors[itemIndex][feature] = random.nextGaussian() * NOISE;
      }
    }
  }

  //TODO: needs optimization
  private double getMu(int i) {
    return mu0 * Math.pow(decayFactor, i - 1) * Math.pow(i + stepOffset, forgettingExponent);
  }

  @Override
  public Factorization factorize() throws TasteException {
    initialize();

    if (logger.isInfoEnabled()) {
      logger.info("starting to compute the factorization...");
    }

    for (epoch = 1; epoch <= numEpochs; epoch++) {
      shuffler.stage();

      final double mu = getMu(epoch);
      int subSize = shuffler.size() / numThreads + 1;

      ExecutorService executor=Executors.newFixedThreadPool(numThreads);

      try {
        for (int t = 0; t < numThreads; t++) {
          final int iStart = t * subSize;
          final int iEnd = Math.min((t + 1) * subSize, shuffler.size());

          executor.execute(new Runnable() {
            @Override
            public void run() {
              for (int i = iStart; i < iEnd; i++) {
                update(shuffler.get(i), mu);
              }
            }
          });
        }
      } finally {
        executor.shutdown();
        shuffler.shuffle();

        try {
          boolean terminated = executor.awaitTermination(numEpochs * shuffler.size(), TimeUnit.MICROSECONDS);
          if (!terminated) {
            logger.error("subtasks takes forever, return anyway");
          }
        } catch (InterruptedException e) {
          throw new TasteException("waiting fof termination interrupted", e);
        }
      }

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

  /** TODO: this is the vanilla sgd by Tacaks 2009, I speculate that using scaling technique proposed in:
   * Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent section 5, page 6
   * can be beneficial in term s of both speed and accuracy.
   *
   * Tacaks' method doesn't calculate gradient of regularization correctly, which has non-zero elements everywhere of
   * the matrix. While Tacaks' method can only updates a single row/column, if one user has a lot of recommendation,
   * her vector will be more affected by regularization using an isolated scaling factor for both user vectors and
   * item vectors can remove this issue without inducing more update cost it even reduces it a bit by only performing
   * one addition and one multiplication.
   *
   * BAD SIDE1: the scaling factor decreases fast, it has to be scaled up from time to time before dropped to zero or
   *            caused roundoff error
   * BAD SIDE2: no body experiment on it before, and people generally use very small lambda
   *            so it's impact on accuracy may still be unknown.
   * BAD SIDE3: don't know how to make it work for L1-regularization or
   *            "pseudorank?" (sum of singular values)-regularization */
  protected void update(Preference preference, double mu) {
    int userIndex = userIndex(preference.getUserID());
    int itemIndex = itemIndex(preference.getItemID());

    double[] userVector = userVectors[userIndex];
    double[] itemVector = itemVectors[itemIndex];

    double prediction = dot(userVector, itemVector);
    double err = preference.getValue() - prediction;

    // adjust features
    for (int k = FEATURE_OFFSET; k < rank; k++) {
      double userFeature = userVector[k];
      double itemFeature = itemVector[k];

      userVector[k] += mu * (err * itemFeature - lambda * userFeature);
      itemVector[k] += mu * (err * userFeature - lambda * itemFeature);
    }

    // adjust user and item bias
    userVector[USER_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * userVector[USER_BIAS_INDEX]);
    itemVector[ITEM_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * itemVector[ITEM_BIAS_INDEX]);
  }

  private double dot(double[] userVector, double[] itemVector) {
    double sum = 0;
    for (int k = 0; k < rank; k++) {
      sum += userVector[k] * itemVector[k];
    }
    return sum;
  }
}