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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


final class MockRecommender implements Recommender {

  private final AtomicInteger recommendCount;

  MockRecommender(AtomicInteger recommendCount) {
    this.recommendCount = recommendCount;
  }

  @Override
  public List<RecommendedItem> recommend(Comparable<?> userID, int howMany) {
    recommendCount.incrementAndGet();
    return Collections.<RecommendedItem>singletonList(
        new GenericRecommendedItem("1", 1.0));
  }

  @Override
  public List<RecommendedItem> recommend(Comparable<?> userID,
                                         int howMany,
                                         Rescorer<Comparable<?>> rescorer) {
    return recommend(userID, howMany);
  }

  @Override
  public double estimatePreference(Comparable<?> userID, Comparable<?> itemID) {
    recommendCount.incrementAndGet();
    return 0.0;
  }

  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, double value) {
    // do nothing
  }

  @Override
  public void removePreference(Comparable<?> userID, Comparable<?> itemID) {
    // do nothing
  }

  @Override
  public DataModel getDataModel() {
    User user1 = new GenericUser("1", Collections.<Preference>emptyList());
    User user2 = new GenericUser("2", Collections.<Preference>emptyList());
    User user3 = new GenericUser("3", Collections.<Preference>emptyList());
    List<User> users = new ArrayList<User>(3);
    users.add(user1);
    users.add(user2);
    users.add(user3);
    return new GenericDataModel(users);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
