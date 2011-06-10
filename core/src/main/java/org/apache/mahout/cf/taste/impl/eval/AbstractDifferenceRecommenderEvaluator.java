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

package org.apache.mahout.cf.taste.impl.eval;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * Abstract superclass of a couple implementations, providing shared functionality.
 */
public abstract class AbstractDifferenceRecommenderEvaluator implements RecommenderEvaluator {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractDifferenceRecommenderEvaluator.class);
  
  private final Random random;
  private float maxPreference;
  private float minPreference;
  
  protected AbstractDifferenceRecommenderEvaluator() {
    random = RandomUtils.getRandom();
    maxPreference = Float.NaN;
    minPreference = Float.NaN;
  }
  
  @Override
  public final float getMaxPreference() {
    return maxPreference;
  }
  
  @Override
  public final void setMaxPreference(float maxPreference) {
    this.maxPreference = maxPreference;
  }
  
  @Override
  public final float getMinPreference() {
    return minPreference;
  }
  
  @Override
  public final void setMinPreference(float minPreference) {
    this.minPreference = minPreference;
  }
  
  @Override
  public double evaluate(RecommenderBuilder recommenderBuilder,
                         DataModelBuilder dataModelBuilder,
                         DataModel dataModel,
                         double trainingPercentage,
                         double evaluationPercentage) throws TasteException {
    Preconditions.checkNotNull(recommenderBuilder);
    Preconditions.checkNotNull(dataModel);
    Preconditions.checkArgument(trainingPercentage >= 0.0 && trainingPercentage <= 1.0,
      "Invalid trainingPercentage: " + trainingPercentage);
    Preconditions.checkArgument(evaluationPercentage >= 0.0 && evaluationPercentage <= 1.0,
      "Invalid evaluationPercentage: " + evaluationPercentage);

    log.info("Beginning evaluation using {} of {}", trainingPercentage, dataModel);
    
    int numUsers = dataModel.getNumUsers();
    FastByIDMap<PreferenceArray> trainingUsers = new FastByIDMap<PreferenceArray>(
        1 + (int) (evaluationPercentage * numUsers));
    FastByIDMap<PreferenceArray> testUserPrefs = new FastByIDMap<PreferenceArray>(
        1 + (int) (evaluationPercentage * numUsers));
    
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.nextLong();
      if (random.nextDouble() < evaluationPercentage) {
        processOneUser(trainingPercentage, trainingUsers, testUserPrefs, userID, dataModel);
      }
    }
    
    DataModel trainingModel = dataModelBuilder == null ? new GenericDataModel(trainingUsers)
        : dataModelBuilder.buildDataModel(trainingUsers);
    
    Recommender recommender = recommenderBuilder.buildRecommender(trainingModel);
    
    double result = getEvaluation(testUserPrefs, recommender);
    log.info("Evaluation result: {}", result);
    return result;
  }
  
  private void processOneUser(double trainingPercentage,
                              FastByIDMap<PreferenceArray> trainingUsers,
                              FastByIDMap<PreferenceArray> testUserPrefs,
                              long userID,
                              DataModel dataModel) throws TasteException {
    List<Preference> trainingPrefs = null;
    List<Preference> testPrefs = null;
    PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      Preference newPref = new GenericPreference(userID, prefs.getItemID(i), prefs.getValue(i));
      if (random.nextDouble() < trainingPercentage) {
        if (trainingPrefs == null) {
          trainingPrefs = Lists.newArrayListWithCapacity(3);
        }
        trainingPrefs.add(newPref);
      } else {
        if (testPrefs == null) {
          testPrefs = Lists.newArrayListWithCapacity(3);
        }
        testPrefs.add(newPref);
      }
    }
    if (trainingPrefs != null) {
      trainingUsers.put(userID, new GenericUserPreferenceArray(trainingPrefs));
      if (testPrefs != null) {
        testUserPrefs.put(userID, new GenericUserPreferenceArray(testPrefs));
      }
    }
  }
  
  private float capEstimatedPreference(float estimate) {
    if (estimate > maxPreference) {
      return maxPreference;
    }
    if (estimate < minPreference) {
      return minPreference;
    }
    return estimate;
  }
  
  private double getEvaluation(FastByIDMap<PreferenceArray> testUserPrefs, Recommender recommender)
    throws TasteException {
    reset();
    Collection<Callable<Void>> estimateCallables = Lists.newArrayList();
    AtomicInteger noEstimateCounter = new AtomicInteger();
    for (Map.Entry<Long,PreferenceArray> entry : testUserPrefs.entrySet()) {
      estimateCallables.add(
          new PreferenceEstimateCallable(recommender, entry.getKey(), entry.getValue(), noEstimateCounter));
    }
    log.info("Beginning evaluation of {} users", estimateCallables.size());
    execute(estimateCallables, noEstimateCounter);
    return computeFinalEvaluation();
  }
  
  protected static void execute(Collection<Callable<Void>> callables, AtomicInteger noEstimateCounter)
    throws TasteException {

    callables = wrapWithStatsCallables(callables, noEstimateCounter);
    int numProcessors = Runtime.getRuntime().availableProcessors();
    ExecutorService executor = Executors.newFixedThreadPool(numProcessors);
    log.info("Starting timing of {} tasks in {} threads", callables.size(), numProcessors);
    try {
      List<Future<Void>> futures = executor.invokeAll(callables);
      // Go look for exceptions here, really
      for (Future<Void> future : futures) {
        future.get();
      }
    } catch (InterruptedException ie) {
      throw new TasteException(ie);
    } catch (ExecutionException ee) {
      throw new TasteException(ee.getCause());
    }
    executor.shutdown();
  }
  
  private static Collection<Callable<Void>> wrapWithStatsCallables(Collection<Callable<Void>> callables,
                                                                   AtomicInteger noEstimateCounter) {
    int size = callables.size();
    Collection<Callable<Void>> wrapped = Lists.newArrayList();
    int count = 0;
    RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
    for (Callable<Void> callable : callables) {
      boolean logStats = count++ % 1000 == 0; // log every 100 or so iterations
      wrapped.add(new StatsCallable(callable, logStats, timing, noEstimateCounter));
    }
    return wrapped;
  }
  
  protected abstract void reset();
  
  protected abstract void processOneEstimate(float estimatedPreference, Preference realPref);
  
  protected abstract double computeFinalEvaluation();
  
  public final class PreferenceEstimateCallable implements Callable<Void> {
    
    private final Recommender recommender;
    private final long testUserID;
    private final PreferenceArray prefs;
    private final AtomicInteger noEstimateCounter;
    
    public PreferenceEstimateCallable(Recommender recommender,
                                      long testUserID,
                                      PreferenceArray prefs,
                                      AtomicInteger noEstimateCounter) {
      this.recommender = recommender;
      this.testUserID = testUserID;
      this.prefs = prefs;
      this.noEstimateCounter = noEstimateCounter;
    }
    
    @Override
    public Void call() throws TasteException {
      for (Preference realPref : prefs) {
        float estimatedPreference = Float.NaN;
        try {
          estimatedPreference = recommender.estimatePreference(testUserID, realPref.getItemID());
        } catch (NoSuchUserException nsue) {
          // It's possible that an item exists in the test data but not training data in which case
          // NSEE will be thrown. Just ignore it and move on.
          log.info("User exists in test data but not training data: {}", testUserID);
        } catch (NoSuchItemException nsie) {
          log.info("Item exists in test data but not training data: {}", realPref.getItemID());
        }
        if (Float.isNaN(estimatedPreference)) {
          noEstimateCounter.incrementAndGet();
        } else {
          estimatedPreference = capEstimatedPreference(estimatedPreference);
          processOneEstimate(estimatedPreference, realPref);
        }
      }
      return null;
    }
    
  }
  
  private static final class StatsCallable implements Callable<Void> {
    
    private final Callable<Void> delegate;
    private final boolean logStats;
    private final RunningAverageAndStdDev timing;
    private final AtomicInteger noEstimateCounter;
    
    private StatsCallable(Callable<Void> delegate,
                          boolean logStats,
                          RunningAverageAndStdDev timing,
                          AtomicInteger noEstimateCounter) {
      this.delegate = delegate;
      this.logStats = logStats;
      this.timing = timing;
      this.noEstimateCounter = noEstimateCounter;
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
        log.info("Average time per recommendation: {}ms", average);
        long totalMemory = runtime.totalMemory();
        long memory = totalMemory - runtime.freeMemory();
        log.info("Approximate memory used: {}MB / {}MB", memory / 1000000L, totalMemory / 1000000L);
        log.info("Unable to recommend in {} cases", noEstimateCounter.get());
      }
      return null;
    }
  }
  
}
