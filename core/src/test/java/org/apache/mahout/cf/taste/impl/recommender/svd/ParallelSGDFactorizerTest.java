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

import java.util.Arrays;
import java.util.List;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakLingering;
import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.svd.ParallelSGDFactorizer.PreferenceShuffler;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ParallelSGDFactorizerTest extends TasteTestCase {

  protected DataModel dataModel;

  protected int rank;
  protected double lambda;
  protected int numIterations;

  private RandomWrapper random = (RandomWrapper) RandomUtils.getRandom();

  protected Factorizer factorizer;
  protected SVDRecommender svdRecommender;

  private static final Logger logger = LoggerFactory.getLogger(ParallelSGDFactorizerTest.class);

  private Matrix randomMatrix(int numRows, int numColumns, double range) {
    double[][] data = new double[numRows][numColumns];
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numColumns; j++) {
        double sqrtUniform = random.nextDouble();
        data[i][j] = sqrtUniform * range;
      }
    }
    return new DenseMatrix(data);
  }

  private void normalize(Matrix source, final double range) {
    final double max = source.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector column) {
        return column.maxValue();
      }
    }).maxValue();

    final double min = source.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector column) {
        return column.minValue();
      }
    }).minValue();

    source.assign(new DoubleFunction() {
      @Override
      public double apply(double value) {
        return (value - min) * range / (max - min);
      }
    });
  }

  public void setUpSyntheticData() throws Exception {

    int numUsers = 2000;
    int numItems = 1000;
    double sparsity = 0.5;

    this.rank = 20;
    this.lambda = 0.000000001;
    this.numIterations = 100;

    Matrix users = randomMatrix(numUsers, rank, 1);
    Matrix items = randomMatrix(rank, numItems, 1);
    Matrix ratings = users.times(items);
    normalize(ratings, 5);

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();
    for (int userIndex = 0; userIndex < numUsers; userIndex++) {
      List<Preference> row= Lists.newArrayList();
      for (int itemIndex = 0; itemIndex < numItems; itemIndex++) {
        if (random.nextDouble() <= sparsity) {
          row.add(new GenericPreference(userIndex, itemIndex, (float) ratings.get(userIndex, itemIndex)));
        }
      }

      userData.put(userIndex, new GenericUserPreferenceArray(row));
    }

    dataModel = new GenericDataModel(userData);
  }

  public void setUpToyData() throws Exception {
    this.rank = 3;
    this.lambda = 0.01;
    this.numIterations = 1000;

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();

    userData.put(1L, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(1L, 1L, 5.0f),
        new GenericPreference(1L, 2L, 5.0f),
        new GenericPreference(1L, 3L, 2.0f))));

    userData.put(2L, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(2L, 1L, 2.0f),
        new GenericPreference(2L, 3L, 3.0f),
        new GenericPreference(2L, 4L, 5.0f))));

    userData.put(3L, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(3L, 2L, 5.0f),
        new GenericPreference(3L, 4L, 3.0f))));

    userData.put(4L, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(4L, 1L, 3.0f),
        new GenericPreference(4L, 4L, 5.0f))));
    dataModel = new GenericDataModel(userData);
  }

  @Test
  public void testPreferenceShufflerWithSyntheticData() throws Exception {
    setUpSyntheticData();

    ParallelSGDFactorizer.PreferenceShuffler shuffler = new PreferenceShuffler(dataModel);
    shuffler.shuffle();
    shuffler.stage();

    FastByIDMap<FastByIDMap<Boolean>> checked = new FastByIDMap<FastByIDMap<Boolean>>();

    for (int i = 0; i < shuffler.size(); i++) {
      Preference pref=shuffler.get(i);

      float value = dataModel.getPreferenceValue(pref.getUserID(), pref.getItemID());
      assertEquals(pref.getValue(), value, 0.0);
      if (!checked.containsKey(pref.getUserID())) {
        checked.put(pref.getUserID(), new FastByIDMap<Boolean>());
      }

      assertNull(checked.get(pref.getUserID()).get(pref.getItemID()));

      checked.get(pref.getUserID()).put(pref.getItemID(), true);
    }

    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    int index=0;
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      PreferenceArray preferencesFromUser = dataModel.getPreferencesFromUser(userID);
      for (Preference preference : preferencesFromUser) {
        assertTrue(checked.get(preference.getUserID()).get(preference.getItemID()));
        index++;
      }
    }
    assertEquals(index, shuffler.size());
  }

  @ThreadLeakLingering(linger = 1000)
  @Test
  public void testFactorizerWithToyData() throws Exception {

    setUpToyData();

    long start = System.currentTimeMillis();

    factorizer = new ParallelSGDFactorizer(dataModel, rank, lambda, numIterations, 0.01, 1, 0, 0);

    Factorization factorization = factorizer.factorize();

    long duration = System.currentTimeMillis() - start;

    /* a hold out test would be better, but this is just a toy example so we only check that the
     * factorization is close to the original matrix */
    RunningAverage avg = new FullRunningAverage();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    LongPrimitiveIterator itemIDs;

    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      for (Preference pref : dataModel.getPreferencesFromUser(userID)) {
        double rating = pref.getValue();
        Vector userVector = new DenseVector(factorization.getUserFeatures(userID));
        Vector itemVector = new DenseVector(factorization.getItemFeatures(pref.getItemID()));
        double estimate = userVector.dot(itemVector);
        double err = rating - estimate;

        avg.addDatum(err * err);
      }
    }

    double sum = 0.0;

    userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      Vector userVector = new DenseVector(factorization.getUserFeatures(userID));
      double regularization = userVector.dot(userVector);
      sum += regularization;
    }

    itemIDs = dataModel.getItemIDs();
    while (itemIDs.hasNext()) {
      long itemID = itemIDs.nextLong();
      Vector itemVector = new DenseVector(factorization.getUserFeatures(itemID));
      double regularization = itemVector.dot(itemVector);
      sum += regularization;
    }

    double rmse = Math.sqrt(avg.getAverage());
    double loss = avg.getAverage() / 2 + lambda / 2 * sum;
    logger.info("RMSE: " + rmse + ";\tLoss: " + loss + ";\tTime Used: " + duration);
    assertTrue(rmse < 0.2);
  }

  @ThreadLeakLingering(linger = 1000)
  @Test
  public void testRecommenderWithToyData() throws Exception {

    setUpToyData();

    factorizer = new ParallelSGDFactorizer(dataModel, rank, lambda, numIterations, 0.01, 1, 0,0);
    svdRecommender = new SVDRecommender(dataModel, factorizer);

    /* a hold out test would be better, but this is just a toy example so we only check that the
     * factorization is close to the original matrix */
    RunningAverage avg = new FullRunningAverage();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      for (Preference pref : dataModel.getPreferencesFromUser(userID)) {
        double rating = pref.getValue();
        double estimate = svdRecommender.estimatePreference(userID, pref.getItemID());
        double err = rating - estimate;
        avg.addDatum(err * err);
      }
    }

    double rmse = Math.sqrt(avg.getAverage());
    logger.info("rmse: " + rmse);
    assertTrue(rmse < 0.2);
  }

  @Test
  public void testFactorizerWithWithSyntheticData() throws Exception {

    setUpSyntheticData();

    long start = System.currentTimeMillis();

    factorizer = new ParallelSGDFactorizer(dataModel, rank, lambda, numIterations, 0.01, 1, 0, 0);

    Factorization factorization = factorizer.factorize();

    long duration = System.currentTimeMillis() - start;

    /* a hold out test would be better, but this is just a toy example so we only check that the
     * factorization is close to the original matrix */
    RunningAverage avg = new FullRunningAverage();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    LongPrimitiveIterator itemIDs;

    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      for (Preference pref : dataModel.getPreferencesFromUser(userID)) {
        double rating = pref.getValue();
        Vector userVector = new DenseVector(factorization.getUserFeatures(userID));
        Vector itemVector = new DenseVector(factorization.getItemFeatures(pref.getItemID()));
        double estimate = userVector.dot(itemVector);
        double err = rating - estimate;

        avg.addDatum(err * err);
      }
    }

    double sum = 0.0;

    userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      Vector userVector = new DenseVector(factorization.getUserFeatures(userID));
      double regularization=userVector.dot(userVector);
      sum += regularization;
    }

    itemIDs = dataModel.getItemIDs();
    while (itemIDs.hasNext()) {
      long itemID = itemIDs.nextLong();
      Vector itemVector = new DenseVector(factorization.getUserFeatures(itemID));
      double regularization = itemVector.dot(itemVector);
      sum += regularization;
    }

    double rmse = Math.sqrt(avg.getAverage());
    double loss = avg.getAverage() / 2 + lambda / 2 * sum;
    logger.info("RMSE: " + rmse + ";\tLoss: " + loss + ";\tTime Used: " + duration + "ms");
    assertTrue(rmse < 0.2);
  }

  @Test
  public void testRecommenderWithSyntheticData() throws Exception {

    setUpSyntheticData();

    factorizer= new ParallelSGDFactorizer(dataModel, rank, lambda, numIterations, 0.01, 1, 0, 0);
    svdRecommender = new SVDRecommender(dataModel, factorizer);

    /* a hold out test would be better, but this is just a toy example so we only check that the
     * factorization is close to the original matrix */
    RunningAverage avg = new FullRunningAverage();
    LongPrimitiveIterator userIDs = dataModel.getUserIDs();
    while (userIDs.hasNext()) {
      long userID = userIDs.nextLong();
      for (Preference pref : dataModel.getPreferencesFromUser(userID)) {
        double rating = pref.getValue();
        double estimate = svdRecommender.estimatePreference(userID, pref.getItemID());
        double err = rating - estimate;
        avg.addDatum(err * err);
      }
    }

    double rmse = Math.sqrt(avg.getAverage());
    logger.info("rmse: " + rmse);
    assertTrue(rmse < 0.2);
  }
}