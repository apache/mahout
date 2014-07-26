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

import java.util.Arrays;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;
import org.easymock.EasyMock;
import org.junit.Test;

public class BPROptimizerTest extends TasteTestCase {

  /**
   * rating-matrix
   * 
   *          burger  hotdog  berries  icecream
   *  dog       5       5        2        -
   *  rabbit    2       -        3        5
   *  cow       -       5        -        3
   *  donkey    3       -        -        5
   */
  @Test
  public void testFindSolution() throws TasteException {
    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();

    userData.put(
        1L,
        new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(1L,
            1L, 5.0f), new GenericPreference(1L, 2L, 5.0f),
            new GenericPreference(1L, 3L, 2.0f))));

    userData.put(
        2L,
        new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(2L,
            1L, 2.0f), new GenericPreference(2L, 3L, 3.0f),
            new GenericPreference(2L, 4L, 5.0f))));

    userData.put(
        3L,
        new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(3L,
            2L, 5.0f), new GenericPreference(3L, 4L, 3.0f))));

    userData.put(
        4L,
        new GenericUserPreferenceArray(Arrays.asList(new GenericPreference(4L,
            1L, 3.0f), new GenericPreference(4L, 4L, 5.0f))));

    DataModel dataModel = new GenericDataModel(userData);
    AbstractOptimizer optimizer = new BPROptimizer(dataModel, 500, 0.05,
        0.0025d, 0.00025d, 0.0d, 0.1d);

    optimizer.findSolution();
    SlimSolution slim = optimizer.getSlimSolution();
    Matrix itemWeights = slim.getItemWeights();

    double[][] convergedSolution = new double[][] {
        { -2.683, -2.524, -1.916, -2.917 }, { 0, -3.109, 1.587, -0.713 },
        { -2.465, 0, -0.74, -1.757 }, { 0.211, -1.282, 0, -3.375 },
        { -0.241, -1.609, -2.958, 0 }, };

    for (int i = 0; i < itemWeights.numRows(); i++) {
      for (int j = 0; j < itemWeights.numCols() - 1; j++) {
        assertEquals(convergedSolution[i][j], itemWeights.get(i, j + 1), 0.3);
      }
    }
  }

  @Test
  public void testPrepareTraining() throws TasteException {
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    LongPrimitiveIterator iterator = EasyMock
        .createMock(LongPrimitiveIterator.class);
    PreferenceArray prefs = EasyMock.createMock(PreferenceArray.class);
    final int numIterations = 10;
    final long mockUser = 1L;
    final long mockItem = 1L;
    final int events = 1;

    EasyMock.expect(iterator.hasNext()).andReturn(true);
    EasyMock.expect(iterator.hasNext()).andReturn(false);
    EasyMock.expect(iterator.hasNext()).andReturn(true);
    EasyMock.expect(iterator.hasNext()).andReturn(false);
    EasyMock.expect(iterator.next()).andReturn(mockUser);
    EasyMock.expect(iterator.nextLong()).andReturn(mockItem);
    EasyMock.replay(iterator);

    EasyMock.expect(dataModel.getItemIDs()).andReturn(iterator);
    EasyMock.expect(dataModel.getItemIDs()).andReturn(iterator);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);
    EasyMock.expect(dataModel.getPreferencesForItem(mockItem)).andReturn(prefs);
    EasyMock.expect(prefs.length()).andReturn(events);
    EasyMock.replay(dataModel);

    AbstractOptimizer optimizer = new BPROptimizer(dataModel, numIterations);
    optimizer.prepareTraining();

    EasyMock.verify(dataModel);
  }
  
}
