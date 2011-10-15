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

package org.apache.mahout.cf.taste.impl.recommender.slopeone;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.model.DataModel;
import org.junit.Test;

/** Tests {@link MemoryDiffStorage}. */
public final class MemoryDiffStorageTest extends TasteTestCase {

  @Test
  public void testRecommendableIDsVariedWeighted() throws Exception {
    DataModel model = getDataModelVaried();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.WEIGHTED, Long.MAX_VALUE);
    FastIDSet recommendableItemIDs = storage.getRecommendableItemIDs(1);
    assertEquals(3, recommendableItemIDs.size());
    assertTrue(recommendableItemIDs.contains(1));
    recommendableItemIDs = storage.getRecommendableItemIDs(2);
    assertEquals(2, recommendableItemIDs.size());
    assertTrue(recommendableItemIDs.contains(2));
    assertTrue(recommendableItemIDs.contains(3));
    
    recommendableItemIDs = storage.getRecommendableItemIDs(3);
    assertEquals(1, recommendableItemIDs.size());
    assertTrue(recommendableItemIDs.contains(3));
    
    recommendableItemIDs = storage.getRecommendableItemIDs(4);
    assertEquals(0, recommendableItemIDs.size());
    // the last item has only one recommendation, and so only 4 items are usable
    recommendableItemIDs = storage.getRecommendableItemIDs(5);
    assertEquals(0, recommendableItemIDs.size());
  }
  
  @Test
  public void testRecommendableIDsPockedUnweighted() throws Exception {
    DataModel model = getDataModelPocked();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MAX_VALUE);
    FastIDSet recommendableItemIDs = storage.getRecommendableItemIDs(1);
    assertEquals(0, recommendableItemIDs.size());
    recommendableItemIDs = storage.getRecommendableItemIDs(2);
    assertEquals(1, recommendableItemIDs.size());
    recommendableItemIDs = storage.getRecommendableItemIDs(3);
    assertEquals(0, recommendableItemIDs.size());
    
    recommendableItemIDs = storage.getRecommendableItemIDs(4);
    assertEquals(0, recommendableItemIDs.size());
    
  }
  
  @Test (expected=NoSuchUserException.class)
  public void testUnRecommendableID() throws Exception {
    DataModel model = getDataModel();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.WEIGHTED, Long.MAX_VALUE);
    storage.getRecommendableItemIDs(0);
  }
  
  @Test
  public void testGetDiff() throws Exception {
    DataModel model = getDataModel();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MAX_VALUE);
    RunningAverage average = storage.getDiff(1, 2);
    assertEquals(0.23333333333333334, average.getAverage(), EPSILON);
    assertEquals(3, average.getCount());
  }
  
  @Test
  public void testAdd() throws Exception {
    DataModel model = getDataModel();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MAX_VALUE);
    
    RunningAverage average1 = storage.getDiff(0, 2);
    assertEquals(0.1, average1.getAverage(), EPSILON);
    assertEquals(3, average1.getCount());
    
    RunningAverage average2 = storage.getDiff(1, 2);
    assertEquals(0.23333332935969034, average2.getAverage(), EPSILON);
    assertEquals(3, average2.getCount());
    
    storage.addItemPref(1, 2, 0.8f);
    
    average1 = storage.getDiff(0, 2);
    assertEquals(0.25, average1.getAverage(), EPSILON);
    assertEquals(4, average1.getCount());
    
    average2 = storage.getDiff(1, 2);
    assertEquals(0.3, average2.getAverage(), EPSILON);
    assertEquals(4, average2.getCount());
  }
  
  @Test
  public void testUpdate() throws Exception {
    DataModel model = getDataModel();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MAX_VALUE);
    
    RunningAverage average = storage.getDiff(1, 2);
    assertEquals(0.23333332935969034, average.getAverage(), EPSILON);
    assertEquals(3, average.getCount());
    
    storage.updateItemPref(1, 0.5f);
    
    average = storage.getDiff(1, 2);
    assertEquals(0.06666666666666668, average.getAverage(), EPSILON);
    assertEquals(3, average.getCount());
  }
  
  @Test
  public void testRemove() throws Exception {
    DataModel model = getDataModel();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MAX_VALUE);
    
    RunningAverage average1 = storage.getDiff(0, 2);
    assertEquals(0.1, average1.getAverage(), EPSILON);
    assertEquals(3, average1.getCount());
    
    RunningAverage average2 = storage.getDiff(1, 2);
    assertEquals(0.23333332935969034, average2.getAverage(), EPSILON);
    assertEquals(3, average2.getCount());
    
    storage.removeItemPref(4, 2, 0.8f);
    
    average1 = storage.getDiff(0, 2);
    assertEquals(0.1, average1.getAverage(), EPSILON);
    assertEquals(2, average1.getCount());
    
    average2 = storage.getDiff(1, 2);
    assertEquals(0.1, average2.getAverage(), EPSILON);
    assertEquals(2, average2.getCount());
  }
  
  @Test (expected=UnsupportedOperationException.class)
  public void testUpdateWeighted() throws Exception {
    DataModel model = getDataModelVaried();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.WEIGHTED, Long.MAX_VALUE);
    
    storage.updateItemPref(2, 0.8f);
  }
  
  @Test
  public void testRemovePref() throws Exception {
    double eps = 0.0001;
    DataModel model = getDataModelPocked();
    MemoryDiffStorage storage = new MemoryDiffStorage(model, Weighting.WEIGHTED, Long.MAX_VALUE);
    
    RunningAverageAndStdDev average = (RunningAverageAndStdDev) storage.getDiff(0, 1);
    assertEquals(-0.033333, average.getAverage(), eps);
    assertEquals(0.32145, average.getStandardDeviation(), eps);
    assertEquals(3, average.getCount());

    storage.removeItemPref(2, 1, 0.1f);
    average = (RunningAverageAndStdDev) storage.getDiff(0, 1);
    assertEquals(0.00000001, average.getAverage(), eps);
    assertEquals(0.44721, average.getStandardDeviation(), eps);
    assertEquals(2, average.getCount());
  }
  
  static DataModel getDataModelVaried() {
    return getDataModel(
        new long[] {1, 2, 3, 4, 5},
        new Double[][] {
            {0.2},
            {0.4, 0.5},
            {0.7, 0.1, 0.5},
            {0.7, 0.3, 0.8, 0.1},
            {0.2, 0.3, 0.6, 0.1, 0.3},
        });
  }
  
  static DataModel getDataModelPocked() {
    return getDataModel(
        new long[] {1, 2, 3, 4},
        new Double[][] {
            {0.1, 0.3},
            {0.2},
            {0.4, 0.5},
            {0.7, 0.3, 0.8},
        });
  }
  
  static DataModel getDataModelLarge() {
    return getDataModel(
        new long[] {1, 2, 3, 4, 5, 6, 7},
        new Double[][] {
            {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
            {0.4, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3},
            {0.7, 0.1, 0.5, 0.2, 0.7, 0.8, 0.9},
            {0.7, 0.3, 0.8, 0.1, 0.6, 0.6, 0.6},
            {0.2, 0.3, 0.6, 0.1, 0.3, 0.4, 0.4},
            {0.2, 0.3, 0.6, 0.1, 0.3, 0.4, 0.4},
            {0.2, 0.3, 0.6, 0.1, 0.3, 0.5, 0.5},
        });
  }
  

}
