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

import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.SparseMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ImplicitLinearRegressionFactorizer extends AbstractFactorizer {

  private static final Logger log = LoggerFactory.getLogger(ImplicitLinearRegressionFactorizer.class);
  private final double preventOverfitting;
  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** number of iterations */
  private final int numIterations;
  private final DataModel dataModel;
  /** User singular vector. */
  private double[][] userMatrix;
  /** Item singular vector. */
  private double[][] itemMatrix;
  private Matrix userTransUser;
  private Matrix itemTransItem;
  Collection<Callable<Void>> fVectorCallables;
  private boolean recomputeUserFeatures;
  private RunningAverage avrChange;

  public ImplicitLinearRegressionFactorizer(DataModel dataModel) throws TasteException {
    this(dataModel, 64, 10, 0.1);
  }

  public ImplicitLinearRegressionFactorizer(DataModel dataModel, int numFeatures, int numIterations,
                                            double preventOverfitting) throws TasteException {

    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.numIterations = numIterations;
    this.preventOverfitting = preventOverfitting;
    fVectorCallables = Lists.newArrayList();
    avrChange = new FullRunningAverage();
  }

  @Override
  public Factorization factorize() throws TasteException {
    Random random = RandomUtils.getRandom();
    userMatrix = new double[dataModel.getNumUsers()][numFeatures];
    itemMatrix = new double[dataModel.getNumItems()][numFeatures];

    /* start with the user side */
    recomputeUserFeatures = true;

    double average = getAveragePreference();

    double prefInterval = dataModel.getMaxPreference() - dataModel.getMinPreference();
    double defaultValue = Math.sqrt((average - prefInterval * 0.1) / numFeatures);
    double interval = prefInterval * 0.1 / numFeatures;

    for (int feature = 0; feature < numFeatures; feature++) {
      for (int userIndex = 0; userIndex < dataModel.getNumUsers(); userIndex++) {
        userMatrix[userIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * random.nextDouble();
      }
      for (int itemIndex = 0; itemIndex < dataModel.getNumItems(); itemIndex++) {
        itemMatrix[itemIndex][feature] = defaultValue + (random.nextDouble() - 0.5) * interval * random.nextDouble();
      }
    }
    train();
    return createFactorization(userMatrix, itemMatrix);
  }

  public void train() throws TasteException {
    for (int i = 0; i < numIterations; i++) {
      if (recomputeUserFeatures) {
        LongPrimitiveIterator userIds = dataModel.getUserIDs();
        /* start with calculating X^TX or Y^TX */
        log.info("Calculating Y^TY");
        reCalculateTrans(recomputeUserFeatures);
        log.info("Building callables for users.");
        while (userIds.hasNext()) {
          long userId = userIds.nextLong();
          int useridx = userIndex(userId);
          buildCallables(buildConfidenceMatrixForUser(userId), buildPreferenceVectorForUser(userId), useridx);
        }
        finishProcessing();
      } else {
        LongPrimitiveIterator itemIds = dataModel.getItemIDs();
        /* start with calculating X^TX or Y^TX */
        log.info("Calculating X^TX");
        reCalculateTrans(recomputeUserFeatures);
        log.info("Building callables for items.");
        while (itemIds.hasNext()) {
          long itemId = itemIds.nextLong();
          int itemidx = itemIndex(itemId);
          buildCallables(buildConfidenceMatrixForItem(itemId), buildPreferenceVectorForItem(itemId), itemidx);
        }
        finishProcessing();
      }
    }
  }

  public Matrix buildPreferenceVectorForUser(long realId) throws TasteException {
    Matrix ids = new SparseMatrix(1, dataModel.getNumItems());
    for (Preference pref : dataModel.getPreferencesFromUser(realId)) {
      int itemidx = itemIndex(pref.getItemID());
      ids.setQuick(0, itemidx, pref.getValue());
    }
    return ids;
  }

  private Matrix buildConfidenceMatrixForItem(long itemId) throws TasteException {
    PreferenceArray prefs = dataModel.getPreferencesForItem(itemId);
    Matrix confidenceMatrix = new SparseMatrix(dataModel.getNumUsers(), dataModel.getNumUsers());
    for (Preference pref : prefs) {
      long userId = pref.getUserID();
      int userIdx = userIndex(userId);
      confidenceMatrix.setQuick(userIdx, userIdx, 1);
    }
    return new DiagonalMatrix(confidenceMatrix);
  }

  private Matrix buildConfidenceMatrixForUser(long userId) throws TasteException {
    PreferenceArray prefs = dataModel.getPreferencesFromUser(userId);
    Matrix confidenceMatrix = new SparseMatrix(dataModel.getNumItems(), dataModel.getNumItems());
    for (Preference pref : prefs) {
      long itemId = pref.getItemID();
      int itemIdx = itemIndex(itemId);
      confidenceMatrix.setQuick(itemIdx, itemIdx, 1);
    }
    return new DiagonalMatrix(confidenceMatrix);
  }

  private Matrix buildPreferenceVectorForItem(long realId) throws TasteException {
    Matrix ids = new SparseMatrix(1, dataModel.getNumUsers());
    for (Preference pref : dataModel.getPreferencesForItem(realId)) {
      int useridx = userIndex(pref.getUserID());
      ids.setQuick(0, useridx, pref.getValue());
    }
    return ids;
  }

  private Matrix ones(int size) {
    double[] vector = new double[size];
    for (int i = 0; i < size; i++) {
      vector[i] = 1;
    }
    Matrix ones = new DiagonalMatrix(vector);
    return ones;
  }

  private double getAveragePreference() throws TasteException {
    RunningAverage average = new FullRunningAverage();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      int count = 0;
      PreferenceArray prefs;
      try {
        prefs = dataModel.getPreferencesFromUser(it.nextLong());
        for (Preference pref : prefs) {
          average.addDatum(pref.getValue());
          count++;
        }
      } catch (NoSuchUserException ex) {
        continue;
      }
      /* add the remaining zeros */
      for (int i = 0; i < (dataModel.getNumItems() - count); i++) {
        average.addDatum(0);
      }
    }
    return average.getAverage();
  }

  /**
   * Recalculating Y^TY or X^TX which is needed for further calculations
   * @param recomputeUserFeatures
   */
  public void reCalculateTrans(boolean recomputeUserFeatures) {
    if (!recomputeUserFeatures) {
      Matrix uMatrix = new DenseMatrix(userMatrix);
      userTransUser = uMatrix.transpose().times(uMatrix);
    } else {
      Matrix iMatrix = new DenseMatrix(itemMatrix);
      itemTransItem = iMatrix.transpose().times(iMatrix);
    }
  }

  private synchronized void updateMatrix(int id, Matrix m) {
    double normA = 0;
    double normB = 0;
    double aTb = 0;
    for (int feature = 0; feature < numFeatures; feature++) {
      if (recomputeUserFeatures) {
        normA += userMatrix[id][feature] * userMatrix[id][feature];
        normB += m.get(feature, 0) * m.get(feature, 0);
        aTb += userMatrix[id][feature] * m.get(feature, 0);
        userMatrix[id][feature] = m.get(feature, 0);
      } else {
        normA += itemMatrix[id][feature] * itemMatrix[id][feature];
        normB += m.get(feature, 0) * m.get(feature, 0);
        aTb += itemMatrix[id][feature] * m.get(feature, 0);
        itemMatrix[id][feature] = m.get(feature, 0);
      }
    }
    /* calculating cosine similarity to determine when to stop the algorithm, this could be used to detect convergence */
    double cosine = (aTb) / (Math.sqrt(normA) * Math.sqrt(normB));
    if (Double.isNaN(cosine)) {
      log.info("Cosine similarity is NaN, recomputeUserFeatures=" + recomputeUserFeatures + " id=" + id);
    } else {
      avrChange.addDatum(cosine);
    }
  }

  public void resetCallables() {
    fVectorCallables = Lists.newArrayList();
  }

  private void resetAvrChange() {
    log.info("Avr Change: {}", avrChange.getAverage());
    avrChange = new FullRunningAverage();
  }

  public void buildCallables(Matrix C, Matrix prefVector, int id) throws TasteException {
    fVectorCallables.add(new FeatureVectorCallable(C, prefVector, id));
    if (fVectorCallables.size() % (200 * Runtime.getRuntime().availableProcessors()) == 0) {
      execute(fVectorCallables);
      resetCallables();
    }
  }

  public void finishProcessing() throws TasteException {
    /* run the remaining part */
    if (fVectorCallables != null) {
      execute(fVectorCallables);
    }
    resetCallables();
    if ((recomputeUserFeatures && avrChange.getCount() != userMatrix.length)
        || (!recomputeUserFeatures && avrChange.getCount() != itemMatrix.length)) {
      log.info("Matrix length is not equal to count");
    }
    resetAvrChange();
    recomputeUserFeatures = !recomputeUserFeatures;
  }

  public Matrix identityV(int size) {
    return ones(size);
  }

  void execute(Collection<Callable<Void>> callables) throws TasteException {
    callables = wrapWithStatsCallables(callables);
    int numProcessors = Runtime.getRuntime().availableProcessors();
    ExecutorService executor = Executors.newFixedThreadPool(numProcessors);
    log.info("Starting timing of {} tasks in {} threads", callables.size(), numProcessors);
    try {
      List<Future<Void>> futures = executor.invokeAll(callables);
      //TODO go look for exceptions here, really
      for (Future<Void> future : futures) {
        future.get();
      }
    } catch (InterruptedException ie) {
      log.warn("error in factorization", ie);
    } catch (ExecutionException ee) {
      log.warn("error in factorization", ee);
    }
    executor.shutdown();
  }

  private Collection<Callable<Void>> wrapWithStatsCallables(Collection<Callable<Void>> callables) {
    int size = callables.size();
    Collection<Callable<Void>> wrapped = Lists.newArrayListWithExpectedSize(size);
    int count = 1;
    RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
    for (Callable<Void> callable : callables) {
      boolean logStats = count++ % 1000 == 0;
      wrapped.add(new StatsCallable(callable, logStats, timing));
    }
    return wrapped;
  }

  private class FeatureVectorCallable implements Callable<Void> {

    private final Matrix C;
    private final Matrix prefVector;
    private final int id;

    private FeatureVectorCallable(Matrix C, Matrix prefVector, int id) {
      this.C = C;
      this.prefVector = prefVector;
      this.id = id;
    }

    @Override
      public Void call() throws Exception {
      Matrix XTCX;
      if (recomputeUserFeatures) {
        Matrix I = identityV(dataModel.getNumItems());
        Matrix I2 = identityV(numFeatures);
        Matrix iTi = itemTransItem.clone();
        Matrix itemM = new DenseMatrix(itemMatrix);
        XTCX = iTi.plus(itemM.transpose().times(C.minus(I)).times(itemM));

        Matrix diag = solve(XTCX.plus(I2.times(preventOverfitting)), I2);
        Matrix results = diag.times(itemM.transpose().times(C)).times(prefVector.transpose());
        updateMatrix(id, results);
      } else {
        Matrix I = identityV(dataModel.getNumUsers());
        Matrix I2 = identityV(numFeatures);
        Matrix uTu = userTransUser.clone();
        Matrix userM = new DenseMatrix(userMatrix);
        XTCX = uTu.plus(userM.transpose().times(C.minus(I)).times(userM));

        Matrix diag = solve(XTCX.plus(I2.times(preventOverfitting)), I2);
        Matrix results = diag.times(userM.transpose().times(C)).times(prefVector.transpose());
        updateMatrix(id, results);
      }
      return null;
    }
  }

  private Matrix solve(Matrix A, Matrix y) {
    return new QRDecomposition(A).solve(y);
  }

  private static class StatsCallable implements Callable<Void> {

    private final Callable<Void> delegate;
    private final boolean logStats;
    private final RunningAverageAndStdDev timing;

    private StatsCallable(Callable<Void> delegate, boolean logStats, RunningAverageAndStdDev timing) {
      this.delegate = delegate;
      this.logStats = logStats;
      this.timing = timing;
    }

    @Override
      public Void call() throws Exception {
      long start = System.currentTimeMillis();
      delegate.call();
      long end = System.currentTimeMillis();
      timing.addDatum(end - start);
      if (logStats) {
        Runtime runtime = Runtime.getRuntime();
        int average = (int) timing.getAverage();
        log.info("Average time per task: {}ms", average);
        long totalMemory = runtime.totalMemory();
        long memory = totalMemory - runtime.freeMemory();
        log.info("Approximate memory used: {}MB / {}MB", memory / 1000000L, totalMemory / 1000000L);
      }
      return null;
    }
  }
}
