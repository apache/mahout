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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.easymock.EasyMock;
import org.junit.Test;

import java.util.Collections;

/**
 * Tests {@link AllUnknownItemsCandidateItemsStrategyTest}
 */
public final class AllUnknownItemsCandidateItemsStrategyTest extends TasteTestCase {

  @Test  
  public void testStrategy() throws TasteException {
    FastIDSet allItemIDs = new FastIDSet();
    allItemIDs.addAll(new long[] { 1L, 2L, 3L });

    FastIDSet preferredItemIDs = new FastIDSet(1);
    preferredItemIDs.add(2L);
    
    DataModel dataModel = EasyMock.createMock(DataModel.class);
    EasyMock.expect(dataModel.getNumItems()).andReturn(3);
    EasyMock.expect(dataModel.getItemIDs()).andReturn(allItemIDs.iterator());

    PreferenceArray prefArrayOfUser123 = new GenericUserPreferenceArray(Collections.singletonList(
        new GenericPreference(123L, 2L, 1.0f)));

    CandidateItemsStrategy strategy = new AllUnknownItemsCandidateItemsStrategy();

    EasyMock.replay(dataModel);

    FastIDSet candidateItems = strategy.getCandidateItems(123L, prefArrayOfUser123, dataModel);
    assertEquals(2, candidateItems.size());
    assertTrue(candidateItems.contains(1L));
    assertTrue(candidateItems.contains(3L));

    EasyMock.verify(dataModel);
  }

}
