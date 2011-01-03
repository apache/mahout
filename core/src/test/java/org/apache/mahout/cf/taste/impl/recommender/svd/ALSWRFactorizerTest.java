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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

public class ALSWRFactorizerTest extends TasteTestCase {

  ALSWRFactorizer factorizer;
  DataModel dataModel;

  /**
   *  rating-matrix
   *
   *          burger  hotdog  berries  icecream
   *  dog       5       5        2        -
   *  rabbit    2       -        3        5
   *  cow       -       5        -        3
   *  donkey    3       -        -        5
   */
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();

    userData.put(1l, new GenericUserPreferenceArray(Arrays.asList(new Preference[] {
        new GenericPreference(1l, 1l, 5f),
        new GenericPreference(1l, 2l, 5f),
        new GenericPreference(1l, 3l, 2f) })));

    userData.put(2l, new GenericUserPreferenceArray(Arrays.asList(new Preference[] {
        new GenericPreference(2l, 1l, 2f),
        new GenericPreference(2l, 3l, 3f),
        new GenericPreference(2l, 4l, 5f) })));

    userData.put(3l, new GenericUserPreferenceArray(Arrays.asList(new Preference[] {
        new GenericPreference(3l, 2l, 5f),
        new GenericPreference(3l, 4l, 3f) })));

    userData.put(4l, new GenericUserPreferenceArray(Arrays.asList(new Preference[] {
        new GenericPreference(4l, 1l, 3f),
        new GenericPreference(4l, 4l, 5f) })));

    dataModel = new GenericDataModel(userData);
    factorizer = new ALSWRFactorizer(dataModel, 3, 0.065, 10);
  }

  @Test
  public void setFeatureColumn() throws Exception {
    double[][] matrix = new double[3][3];
    Vector vector = new DenseVector(new double[] { 0.5, 2.0, 1.5 });
    int index = 1;

    factorizer.setFeatureColumn(matrix, index, vector);

    assertEquals(vector.get(0), matrix[index][0], EPSILON);
    assertEquals(vector.get(1), matrix[index][1], EPSILON);
    assertEquals(vector.get(2), matrix[index][2], EPSILON);
  }

  @Test
  public void ratingVector() throws Exception {
    PreferenceArray prefs = dataModel.getPreferencesFromUser(1);

    Vector ratingVector = factorizer.ratingVector(prefs);

    assertEquals(prefs.length(), ratingVector.getNumNondefaultElements());
    assertEquals(prefs.get(0).getValue(), ratingVector.get(0), EPSILON);
    assertEquals(prefs.get(1).getValue(), ratingVector.get(1), EPSILON);
    assertEquals(prefs.get(2).getValue(), ratingVector.get(2), EPSILON);
  }

  @Test
  public void averageRating() throws Exception {
    assertEquals(2.5, factorizer.averateRating(3l), EPSILON);
  }

  @Test
  public void initializeM() throws Exception {
    double[][] M = factorizer.initializeM();

    assertEquals(3.333333333, M[0][0], EPSILON);
    assertEquals(5, M[1][0], EPSILON);
    assertEquals(2.5, M[2][0], EPSILON);
    assertEquals(4.333333333, M[3][0], EPSILON);

    for (int itemIndex = 0; itemIndex < dataModel.getNumItems(); itemIndex++) {
      for (int feature = 1; feature < 3; feature++ ) {
        assertTrue(M[itemIndex][feature] >= 0);
        assertTrue(M[itemIndex][feature] <= 0.1);
      }
    }
  }

  @Test
  public void toyExample() throws Exception {

    SVDRecommender svdRecommender = new SVDRecommender(dataModel, factorizer);

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
    assertTrue(rmse < 0.2d);
  }
}
