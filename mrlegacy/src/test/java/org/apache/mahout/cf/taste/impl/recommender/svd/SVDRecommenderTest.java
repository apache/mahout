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
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.easymock.EasyMock;
import org.junit.Test;

import java.util.List;

public class SVDRecommenderTest extends TasteTestCase {

  @Test
  public void estimatePreference() throws Exception {
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    Factorizer factorizer = EasyMock.createMock(Factorizer.class);
    Factorization factorization = EasyMock.createMock(Factorization.class);

    EasyMock.expect(factorizer.factorize()).andReturn(factorization);
    EasyMock.expect(factorization.getUserFeatures(1L)).andReturn(new double[] { 0.4, 2 });
    EasyMock.expect(factorization.getItemFeatures(5L)).andReturn(new double[] { 1, 0.3 });
    EasyMock.replay(dataModel, factorizer, factorization);

    SVDRecommender svdRecommender = new SVDRecommender(dataModel, factorizer);

    float estimate = svdRecommender.estimatePreference(1L, 5L);
    assertEquals(1, estimate, EPSILON);

    EasyMock.verify(dataModel, factorizer, factorization);
  }

  @Test
  public void recommend() throws Exception {
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    PreferenceArray preferencesFromUser = EasyMock.createMock(PreferenceArray.class);
    CandidateItemsStrategy candidateItemsStrategy = EasyMock.createMock(CandidateItemsStrategy.class);
    Factorizer factorizer = EasyMock.createMock(Factorizer.class);
    Factorization factorization = EasyMock.createMock(Factorization.class);

    FastIDSet candidateItems = new FastIDSet();
    candidateItems.add(5L);
    candidateItems.add(3L);

    EasyMock.expect(factorizer.factorize()).andReturn(factorization);
    EasyMock.expect(dataModel.getPreferencesFromUser(1L)).andReturn(preferencesFromUser);
    EasyMock.expect(candidateItemsStrategy.getCandidateItems(1L, preferencesFromUser, dataModel))
        .andReturn(candidateItems);
    EasyMock.expect(factorization.getUserFeatures(1L)).andReturn(new double[] { 0.4, 2 });
    EasyMock.expect(factorization.getItemFeatures(5L)).andReturn(new double[] { 1, 0.3 });
    EasyMock.expect(factorization.getUserFeatures(1L)).andReturn(new double[] { 0.4, 2 });
    EasyMock.expect(factorization.getItemFeatures(3L)).andReturn(new double[] { 2, 0.6 });

    EasyMock.replay(dataModel, candidateItemsStrategy, factorizer, factorization);

    SVDRecommender svdRecommender = new SVDRecommender(dataModel, factorizer, candidateItemsStrategy);

    List<RecommendedItem> recommendedItems = svdRecommender.recommend(1L, 5);
    assertEquals(2, recommendedItems.size());
    assertEquals(3L, recommendedItems.get(0).getItemID());
    assertEquals(2.0f, recommendedItems.get(0).getValue(), EPSILON);
    assertEquals(5L, recommendedItems.get(1).getItemID());
    assertEquals(1.0f, recommendedItems.get(1).getValue(), EPSILON);

    EasyMock.verify(dataModel, candidateItemsStrategy, factorizer, factorization);
  }
}
