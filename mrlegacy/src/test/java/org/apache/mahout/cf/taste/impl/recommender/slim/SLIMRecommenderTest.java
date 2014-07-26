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

package org.apache.mahout.cf.taste.impl.recommender.slim;

import java.util.List;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Matrix;
import org.easymock.EasyMock;
import org.junit.Test;

public class SLIMRecommenderTest extends TasteTestCase {

  @Test
  public void estimatePreference() throws Exception {
    final long userID = 1L;
    final long itemID = 5L;
    final long prefItemID = 3L;

    DataModel dataModel = EasyMock.createMock(DataModel.class);
    Optimizer optimizer = EasyMock.createMock(Optimizer.class);
    SlimSolution slimSolution = EasyMock.createMock(SlimSolution.class);
        
    estimatePreferenceExpectations(dataModel, optimizer, slimSolution, userID,
        itemID, prefItemID, 0.5);

    EasyMock.expect(optimizer.findSolution()).andReturn(slimSolution);

    EasyMock.replay(dataModel, optimizer, slimSolution);

    SparseLinearMethodsRecommender recommender = new SparseLinearMethodsRecommender(
        dataModel, optimizer);

    float estimate = recommender.estimatePreference(userID, itemID);
    assertEquals(0.5, estimate, EPSILON);

    EasyMock.verify(dataModel, optimizer, slimSolution);
  }

  @Test
  public void recommend() throws Exception {
    final long userID = 1L;
    
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    Optimizer optimizer = EasyMock.createMock(Optimizer.class);
    SlimSolution slimSolution = EasyMock.createMock(SlimSolution.class);
    PreferenceArray prefs = EasyMock.createMock(PreferenceArray.class);
    CandidateItemsStrategy candidateItemsStrategy = EasyMock
        .createMock(CandidateItemsStrategy.class);

    FastIDSet candidateItems = new FastIDSet();
    candidateItems.add(5L);
    candidateItems.add(4L);

    EasyMock.expect(
        candidateItemsStrategy.getCandidateItems(userID, prefs, dataModel,
            false)).andReturn(candidateItems);
    EasyMock.expect(optimizer.findSolution()).andReturn(slimSolution);
    EasyMock.expect(dataModel.getPreferencesFromUser(userID)).andReturn(prefs);

    estimatePreferenceExpectations(dataModel, optimizer, slimSolution, userID,
        5L, 3L, 0.3);
    estimatePreferenceExpectations(dataModel, optimizer, slimSolution, userID,
        4L, 3L, 0.7);
    
    EasyMock.replay(dataModel, slimSolution, optimizer, candidateItemsStrategy);

    SparseLinearMethodsRecommender recommender = new SparseLinearMethodsRecommender(
        dataModel, optimizer, candidateItemsStrategy);

    List<RecommendedItem> recommendedItems = recommender.recommend(userID, 5);
    assertEquals(2, recommendedItems.size());
    assertEquals(4L, recommendedItems.get(0).getItemID());
    assertEquals(0.7f, recommendedItems.get(0).getValue(), EPSILON);
    assertEquals(5L, recommendedItems.get(1).getItemID());
    assertEquals(0.3f, recommendedItems.get(1).getValue(), EPSILON);

    EasyMock.verify(dataModel, optimizer, slimSolution);
  }

  private void estimatePreferenceExpectations(DataModel dataModel,
      Optimizer optimizer, SlimSolution slimSolution, final long userID,
      final long itemID, final long prefItemID, final double estimate)
      throws TasteException, NoSuchItemException {
    final int prefsLength = 1;
    Matrix itemWeights = EasyMock.createMock(Matrix.class);
    PreferenceArray prefs = EasyMock.createMock(PreferenceArray.class);
    
    EasyMock.expect(slimSolution.itemIndex(itemID)).andReturn(0);
    EasyMock.expect(slimSolution.itemIndex(prefItemID)).andReturn(1);
    EasyMock.expect(slimSolution.getItemWeights()).andReturn(itemWeights);
    EasyMock.expect(prefs.length()).andReturn(prefsLength);
    EasyMock.expect(prefs.getItemID(0)).andReturn(prefItemID);
    EasyMock.expect(dataModel.hasPreferenceValues()).andReturn(false);
    EasyMock.expect(itemWeights.getQuick(0, 1)).andReturn(estimate);
    EasyMock.expect(dataModel.getPreferencesFromUser(userID)).andReturn(prefs);
    
    EasyMock.replay(itemWeights, prefs);
  }
  
}
