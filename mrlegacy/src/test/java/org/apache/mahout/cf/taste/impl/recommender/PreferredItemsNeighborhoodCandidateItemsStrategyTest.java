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

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.easymock.EasyMock;
import org.junit.Test;

/**
 * Tests {@link PreferredItemsNeighborhoodCandidateItemsStrategy}
 */
public final class PreferredItemsNeighborhoodCandidateItemsStrategyTest extends TasteTestCase {

  @Test
  public void testStrategy() throws TasteException {
    FastIDSet itemIDsFromUser123 = new FastIDSet();
    itemIDsFromUser123.add(1L);

    FastIDSet itemIDsFromUser456 = new FastIDSet();
    itemIDsFromUser456.add(1L);
    itemIDsFromUser456.add(2L);

    List<Preference> prefs = Lists.newArrayList();
    prefs.add(new GenericPreference(123L, 1L, 1.0f));
    prefs.add(new GenericPreference(456L, 1L, 1.0f));
    PreferenceArray preferencesForItem1 = new GenericItemPreferenceArray(prefs);

    DataModel dataModel = EasyMock.createMock(DataModel.class);
    EasyMock.expect(dataModel.getPreferencesForItem(1L)).andReturn(preferencesForItem1);
    EasyMock.expect(dataModel.getItemIDsFromUser(123L)).andReturn(itemIDsFromUser123);
    EasyMock.expect(dataModel.getItemIDsFromUser(456L)).andReturn(itemIDsFromUser456);

    PreferenceArray prefArrayOfUser123 =
        new GenericUserPreferenceArray(Collections.singletonList(new GenericPreference(123L, 1L, 1.0f)));

    CandidateItemsStrategy strategy = new PreferredItemsNeighborhoodCandidateItemsStrategy();

    EasyMock.replay(dataModel);

    FastIDSet candidateItems = strategy.getCandidateItems(123L, prefArrayOfUser123, dataModel);
    assertEquals(1, candidateItems.size());
    assertTrue(candidateItems.contains(2L));

    EasyMock.verify(dataModel);
  }
  
}
