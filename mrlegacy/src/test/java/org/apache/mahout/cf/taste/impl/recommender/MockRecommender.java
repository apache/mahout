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

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.util.Collection;
import java.util.Collections;
import java.util.List;


final class MockRecommender implements Recommender {

  private final MutableInt recommendCount;

  MockRecommender(MutableInt recommendCount) {
    this.recommendCount = recommendCount;
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany) {
    recommendCount.increment();
    return Collections.<RecommendedItem>singletonList(
        new GenericRecommendedItem(1, 1.0f));
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) {
    return recommend(userID, howMany);
  }

  @Override
  public float estimatePreference(long userID, long itemID) {
    recommendCount.increment();
    return 0.0f;
  }

  @Override
  public void setPreference(long userID, long itemID, float value) {
    // do nothing
  }

  @Override
  public void removePreference(long userID, long itemID) {
    // do nothing
  }

  @Override
  public DataModel getDataModel() {
    return TasteTestCase.getDataModel(
            new long[] {1, 2, 3},
            new Double[][]{{1.0},{2.0},{3.0}});
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
