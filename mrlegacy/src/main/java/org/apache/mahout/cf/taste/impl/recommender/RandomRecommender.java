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

import java.util.Collection;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.RandomUtils;

/**
 * Produces random recommendations and preference estimates. This is likely only useful as a novelty and for
 * benchmarking.
 */
public final class RandomRecommender extends AbstractRecommender {
  
  private final Random random = RandomUtils.getRandom();
  private final float minPref;
  private final float maxPref;
  
  public RandomRecommender(DataModel dataModel) throws TasteException {
    super(dataModel);
    float maxPref = Float.NEGATIVE_INFINITY;
    float minPref = Float.POSITIVE_INFINITY;
    LongPrimitiveIterator userIterator = dataModel.getUserIDs();
    while (userIterator.hasNext()) {
      long userID = userIterator.next();
      PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
      for (int i = 0; i < prefs.length(); i++) {
        float prefValue = prefs.getValue(i);
        if (prefValue < minPref) {
          minPref = prefValue;
        }
        if (prefValue > maxPref) {
          maxPref = prefValue;
        }
      }
    }
    this.minPref = minPref;
    this.maxPref = maxPref;
  }
  
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    DataModel dataModel = getDataModel();
    int numItems = dataModel.getNumItems();
    List<RecommendedItem> result = Lists.newArrayListWithCapacity(howMany);
    while (result.size() < howMany) {
      LongPrimitiveIterator it = dataModel.getItemIDs();
      it.skip(random.nextInt(numItems));
      long itemID = it.next();
      if (dataModel.getPreferenceValue(userID, itemID) == null) {
        result.add(new GenericRecommendedItem(itemID, randomPref()));
      }
    }
    return result;
  }
  
  @Override
  public float estimatePreference(long userID, long itemID) {
    return randomPref();
  }
  
  private float randomPref() {
    return minPref + random.nextFloat() * (maxPref - minPref);
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    getDataModel().refresh(alreadyRefreshed);
  }
  
}
