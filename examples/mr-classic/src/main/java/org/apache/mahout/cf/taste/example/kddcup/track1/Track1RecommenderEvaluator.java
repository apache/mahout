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

package org.apache.mahout.cf.taste.example.kddcup.track1;

import java.io.File;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.example.kddcup.DataFileIterable;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.eval.AbstractDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Attempts to run an evaluation just like that dictated for Yahoo's KDD Cup, Track 1.
 * It will compute the RMSE of a validation data set against the predicted ratings from
 * the training data set.
 */
public final class Track1RecommenderEvaluator extends AbstractDifferenceRecommenderEvaluator {

  private static final Logger log = LoggerFactory.getLogger(Track1RecommenderEvaluator.class);

  private RunningAverage average;
  private final File dataFileDirectory;

  public Track1RecommenderEvaluator(File dataFileDirectory) {
    setMaxPreference(100.0f);
    setMinPreference(0.0f);
    average = new FullRunningAverage();
    this.dataFileDirectory = dataFileDirectory;
  }

  @Override
  public double evaluate(RecommenderBuilder recommenderBuilder,
                         DataModelBuilder dataModelBuilder,
                         DataModel dataModel,
                         double trainingPercentage,
                         double evaluationPercentage) throws TasteException {

    Recommender recommender = recommenderBuilder.buildRecommender(dataModel);

    Collection<Callable<Void>> estimateCallables = Lists.newArrayList();
    AtomicInteger noEstimateCounter = new AtomicInteger();
    for (Pair<PreferenceArray,long[]> userData
        : new DataFileIterable(KDDCupDataModel.getValidationFile(dataFileDirectory))) {
      PreferenceArray validationPrefs = userData.getFirst();
      long userID = validationPrefs.get(0).getUserID();
      estimateCallables.add(
          new PreferenceEstimateCallable(recommender, userID, validationPrefs, noEstimateCounter));
    }

    RunningAverageAndStdDev timing = new FullRunningAverageAndStdDev();
    execute(estimateCallables, noEstimateCounter, timing);

    double result = computeFinalEvaluation();
    log.info("Evaluation result: {}", result);
    return result;
  }

  // Use RMSE scoring:

  @Override
  protected void reset() {
    average = new FullRunningAverage();
  }

  @Override
  protected void processOneEstimate(float estimatedPreference, Preference realPref) {
    double diff = realPref.getValue() - estimatedPreference;
    average.addDatum(diff * diff);
  }

  @Override
  protected double computeFinalEvaluation() {
    return Math.sqrt(average.getAverage());
  }

}
