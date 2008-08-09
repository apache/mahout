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

import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.common.Refreshable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;


final class MockRecommender implements Recommender {

  private final AtomicInteger recommendCount;

  MockRecommender(AtomicInteger recommendCount) {
    this.recommendCount = recommendCount;
  }

  public List<RecommendedItem> recommend(Object userID, int howMany) {
    recommendCount.incrementAndGet();
    return Collections.<RecommendedItem>singletonList(
            new GenericRecommendedItem(new GenericItem<String>("1"), 1.0));
  }

  public List<RecommendedItem> recommend(Object userID,
                                         int howMany,
                                         Rescorer<Item> rescorer) {
    return recommend(userID, howMany);
  }

  public double estimatePreference(Object userID, Object itemID) {
    recommendCount.incrementAndGet();
    return 0.0;
  }

  public void setPreference(Object userID, Object itemID, double value) {
    // do nothing
  }

  public void removePreference(Object userID, Object itemID) {
    // do nothing
  }

  public DataModel getDataModel() {
    User user1 = new GenericUser<String>("1", Collections.<Preference>emptyList());
    User user2 = new GenericUser<String>("2", Collections.<Preference>emptyList());
    User user3 = new GenericUser<String>("3", Collections.<Preference>emptyList());
    List<User> users = new ArrayList<User>(3);
    users.add(user1);
    users.add(user2);
    users.add(user3);
    return new GenericDataModel(users);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
