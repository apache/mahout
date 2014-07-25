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

import java.util.Iterator;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;
import org.easymock.EasyMock;
import org.junit.Test;

public class LeastSquareOptimizerTest extends TasteTestCase {


  @Test
  public void testPrepareTraining() throws TasteException {
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    LongPrimitiveIterator iterator = EasyMock
        .createMock(LongPrimitiveIterator.class);
    PreferenceArray prefs = EasyMock.createMock(PreferenceArray.class);
    SlimSolution slim = EasyMock.createMock(SlimSolution.class);
    Iterator itPrefs = EasyMock.createMock(Iterator.class);
    Preference pref = EasyMock.createMock(Preference.class);
    Matrix slimMatrix = EasyMock.createMock(Matrix.class);
    final int numIterations = 10;
    final long mockUser = 1L;
    final long mockItem = 1L;
    final int events = 1;

    EasyMock.expect(iterator.hasNext()).andReturn(true);
    EasyMock.expect(iterator.hasNext()).andReturn(false);
    EasyMock.expect(iterator.next()).andReturn(mockUser);
    EasyMock.expect(iterator.nextLong()).andReturn(mockItem);
    EasyMock.replay(iterator);
    
    EasyMock.expect(slim.getItemWeights()).andReturn(slimMatrix);

    EasyMock.expect(dataModel.getItemIDs()).andReturn(iterator);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);
    EasyMock.expect(dataModel.getNumItems()).andReturn(events);

    EasyMock.expect(itPrefs.hasNext()).andReturn(true);
    EasyMock.expect(itPrefs.hasNext()).andReturn(false);
    EasyMock.expect(itPrefs.next()).andReturn(pref);
    EasyMock.expect(pref.getItemID()).andReturn(mockItem);
    
    EasyMock.expect(prefs.length()).andReturn(events);
    EasyMock.expect(prefs.iterator()).andReturn(itPrefs);
    EasyMock.expect(prefs.get(0)).andReturn(pref);
    
    EasyMock.replay(dataModel);

    LeastSquareOptimizer optimizer = new LeastSquareOptimizer(dataModel, numIterations, 0);
    optimizer.prepareTraining();
    
    optimizer.slim = slim;

    //double prediction = optimizer.predictWithExclusion(mockUser, mockItem, mockExcludeItem);
    EasyMock.verify(dataModel);
  }

}
