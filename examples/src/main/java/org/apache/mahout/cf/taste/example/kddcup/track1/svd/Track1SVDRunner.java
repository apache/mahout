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

package org.apache.mahout.cf.taste.example.kddcup.track1.svd;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.example.kddcup.DataFileIterable;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.example.kddcup.track1.EstimateConverter;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorizer;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run an SVD factorization of the KDD track1 data.
 *
 * needs at least 6-7GB of memory, tested with -Xms6700M -Xmx6700M
 *
 */
public final class Track1SVDRunner {

  private static final Logger log = LoggerFactory.getLogger(Track1SVDRunner.class);

  private Track1SVDRunner() {
  }

  public static void main(String[] args) throws Exception {

    if (args.length != 2) {
      System.err.println("Necessary arguments: <kddDataFileDirectory> <resultFile>");
      return;
    }

    File dataFileDirectory = new File(args[0]);
    if (!dataFileDirectory.exists() || !dataFileDirectory.isDirectory()) {
      throw new IllegalArgumentException("Bad data file directory: " + dataFileDirectory);
    }

    File resultFile = new File(args[1]);

    /* the knobs to turn */
    int numFeatures = 20;
    int numIterations = 5;
    double learningRate = 0.0001;
    double preventOverfitting = 0.002;
    double randomNoise = 0.0001;


    KDDCupFactorizablePreferences factorizablePreferences =
        new KDDCupFactorizablePreferences(KDDCupDataModel.getTrainingFile(dataFileDirectory));

    Factorizer sgdFactorizer = new ParallelArraysSGDFactorizer(factorizablePreferences, numFeatures, numIterations,
        learningRate, preventOverfitting, randomNoise);

    Factorization factorization = sgdFactorizer.factorize();

    log.info("Estimating validation preferences...");
    int prefsProcessed = 0;
    RunningAverage average = new FullRunningAverage();
    for (Pair<PreferenceArray,long[]> validationPair
        : new DataFileIterable(KDDCupDataModel.getValidationFile(dataFileDirectory))) {
      for (Preference validationPref : validationPair.getFirst()) {
        double estimate = estimatePreference(factorization, validationPref.getUserID(), validationPref.getItemID(),
            factorizablePreferences.getMinPreference(), factorizablePreferences.getMaxPreference());
        double error = validationPref.getValue() - estimate;
        average.addDatum(error * error);
        prefsProcessed++;
        if (prefsProcessed % 100000 == 0) {
          log.info("Computed {} estimations", prefsProcessed);
        }
      }
    }
    log.info("Computed {} estimations, done.", prefsProcessed);

    double rmse = Math.sqrt(average.getAverage());
    log.info("RMSE {}", rmse);

    log.info("Estimating test preferences...");
    OutputStream out = null;
    try {
      out = new BufferedOutputStream(new FileOutputStream(resultFile));

      for (Pair<PreferenceArray,long[]> testPair
          : new DataFileIterable(KDDCupDataModel.getTestFile(dataFileDirectory))) {
        for (Preference testPref : testPair.getFirst()) {
          double estimate = estimatePreference(factorization, testPref.getUserID(), testPref.getItemID(),
              factorizablePreferences.getMinPreference(), factorizablePreferences.getMaxPreference());
          byte result = EstimateConverter.convert(estimate, testPref.getUserID(), testPref.getItemID());
          out.write(result);
        }
      }
    } finally {
      Closeables.close(out, false);
    }
    log.info("wrote estimates to {}, done.", resultFile.getAbsolutePath());
  }

  static double estimatePreference(Factorization factorization, long userID, long itemID, float minPreference,
      float maxPreference) throws NoSuchUserException, NoSuchItemException {
    double[] userFeatures = factorization.getUserFeatures(userID);
    double[] itemFeatures = factorization.getItemFeatures(itemID);
    double estimate = 0;
    for (int feature = 0; feature < userFeatures.length; feature++) {
      estimate += userFeatures[feature] * itemFeatures[feature];
    }
    if (estimate < minPreference) {
      estimate = minPreference;
    } else if (estimate > maxPreference) {
      estimate = maxPreference;
    }
    return estimate;
  }

}
