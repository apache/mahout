/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.similarity.precompute;

import java.io.IOException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

public class MultithreadedBatchItemSimilaritiesTest {

  @Test
  public void lessItemsThanBatchSize() throws Exception {

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<>();
    userData.put(1, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(1, 1, 1),
        new GenericPreference(1, 2, 1), new GenericPreference(1, 3, 1))));
    userData.put(2, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(2, 1, 1),
        new GenericPreference(2, 2, 1), new GenericPreference(2, 4, 1))));

    DataModel dataModel = new GenericDataModel(userData);
    ItemBasedRecommender recommender =
        new GenericItemBasedRecommender(dataModel, new TanimotoCoefficientSimilarity(dataModel));

    BatchItemSimilarities batchSimilarities = new MultithreadedBatchItemSimilarities(recommender, 10);

    batchSimilarities.computeItemSimilarities(1, 1, mock(SimilarItemsWriter.class));
  }

  @Test(expected = IOException.class)
  public void higherDegreeOfParallelismThanBatches() throws Exception {

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<>();
    userData.put(1, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(1, 1, 1),
        new GenericPreference(1, 2, 1), new GenericPreference(1, 3, 1))));
    userData.put(2, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(2, 1, 1),
        new GenericPreference(2, 2, 1), new GenericPreference(2, 4, 1))));

    DataModel dataModel = new GenericDataModel(userData);
    ItemBasedRecommender recommender =
        new GenericItemBasedRecommender(dataModel, new TanimotoCoefficientSimilarity(dataModel));

    BatchItemSimilarities batchSimilarities = new MultithreadedBatchItemSimilarities(recommender, 10);

    // Batch size is 100, so we only get 1 batch from 3 items, but we use a degreeOfParallelism of 2
    batchSimilarities.computeItemSimilarities(2, 1, mock(SimilarItemsWriter.class));
    fail();
  }

  @Test
  public void testCorrectNumberOfOutputSimilarities() throws Exception {
    FastByIDMap<PreferenceArray> userData = new FastByIDMap<>();
    userData.put(1, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(1, 1, 1),
        new GenericPreference(1, 2, 1), new GenericPreference(1, 3, 1))));
    userData.put(2, new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(2, 1, 1),
        new GenericPreference(2, 2, 1), new GenericPreference(2, 4, 1))));

    DataModel dataModel = new GenericDataModel(userData);
    ItemBasedRecommender recommender =
        new GenericItemBasedRecommender(dataModel, new TanimotoCoefficientSimilarity(dataModel));

    BatchItemSimilarities batchSimilarities = new MultithreadedBatchItemSimilarities(recommender, 10, 2);

    int numOutputSimilarities = batchSimilarities.computeItemSimilarities(2, 1, mock(SimilarItemsWriter.class));
    assertEquals(numOutputSimilarities, 10);
  }

}
