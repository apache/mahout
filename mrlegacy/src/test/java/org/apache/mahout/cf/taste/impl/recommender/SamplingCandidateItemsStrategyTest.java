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

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.junit.Test;

import java.util.List;

/**
 * Tests {@link SamplingCandidateItemsStrategy}
 */
public final class SamplingCandidateItemsStrategyTest extends TasteTestCase {

  @Test
  public void testStrategy() throws TasteException {
    List<Preference> prefsOfUser123 = Lists.newArrayList();
    prefsOfUser123.add(new GenericPreference(123L, 1L, 1.0f));

    List<Preference> prefsOfUser456 = Lists.newArrayList();
    prefsOfUser456.add(new GenericPreference(456L, 1L, 1.0f));
    prefsOfUser456.add(new GenericPreference(456L, 2L, 1.0f));

    List<Preference> prefsOfUser789 = Lists.newArrayList();
    prefsOfUser789.add(new GenericPreference(789L, 1L, 0.5f));
    prefsOfUser789.add(new GenericPreference(789L, 3L, 1.0f));

    PreferenceArray prefArrayOfUser123 = new GenericUserPreferenceArray(prefsOfUser123);

    FastByIDMap<PreferenceArray> userData = new FastByIDMap<PreferenceArray>();
    userData.put(123L, prefArrayOfUser123);
    userData.put(456L, new GenericUserPreferenceArray(prefsOfUser456));
    userData.put(789L, new GenericUserPreferenceArray(prefsOfUser789));

    DataModel dataModel = new GenericDataModel(userData);

    CandidateItemsStrategy strategy =
        new SamplingCandidateItemsStrategy(1, 1, 1, dataModel.getNumUsers(), dataModel.getNumItems());

    FastIDSet candidateItems = strategy.getCandidateItems(123L, prefArrayOfUser123, dataModel);
    /* result can be either item2 or item3 or empty */
    assertTrue(candidateItems.size() <= 1);
    assertFalse(candidateItems.contains(1L));
  }
}
