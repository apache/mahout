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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.recommender.ReversingRescorer;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>Tests {@link SlopeOneRecommender}.</p>
 */
public final class SlopeOneRecommenderTest extends TasteTestCase {

  public void testRecommender() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.34803885284992736, firstRecommended.getValue(), EPSILON);
  }

  public void testHowMany() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.6));
    users.add(getUser("test3", 0.4, 0.4, 0.5, 0.9));
    users.add(getUser("test4", 0.1, 0.4, 0.5, 0.8, 0.9, 1.0));
    users.add(getUser("test5", 0.2, 0.3, 0.6, 0.7, 0.1, 0.2));
    DataModel dataModel = new GenericDataModel(users);
    Recommender recommender = new SlopeOneRecommender(dataModel);
    List<RecommendedItem> fewRecommended = recommender.recommend("test1", 2);
    List<RecommendedItem> moreRecommended = recommender.recommend("test1", 4);
    for (int i = 0; i < fewRecommended.size(); i++) {
      assertEquals(fewRecommended.get(i).getItem(), moreRecommended.get(i).getItem());
    }
  }

  public void testRescorer() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.6));
    users.add(getUser("test3", 0.4, 0.4, 0.5, 0.9));
    DataModel dataModel = new GenericDataModel(users);
    Recommender recommender = new SlopeOneRecommender(dataModel);
    List<RecommendedItem> originalRecommended = recommender.recommend("test1", 2);
    List<RecommendedItem> rescoredRecommended =
            recommender.recommend("test1", 2, new ReversingRescorer<Item>());
    assertNotNull(originalRecommended);
    assertNotNull(rescoredRecommended);
    assertEquals(2, originalRecommended.size());
    assertEquals(2, rescoredRecommended.size());
    assertEquals(originalRecommended.get(0).getItem(), rescoredRecommended.get(1).getItem());
    assertEquals(originalRecommended.get(1).getItem(), rescoredRecommended.get(0).getItem());
  }

  public void testEstimatePref() throws Exception {
    Recommender recommender = buildRecommender();
    assertEquals(0.34803885284992736, recommender.estimatePreference("test1", "2"), EPSILON);
  }

  public void testBestRating() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.0, 0.3));
    users.add(getUser("test2", 0.2, 0.3, 0.3));
    users.add(getUser("test3", 0.4, 0.3, 0.5));
    DataModel dataModel = new GenericDataModel(users);
    Recommender recommender = new SlopeOneRecommender(dataModel);
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    // item one should be recommended because it has a greater rating/score
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.2400938676203033, firstRecommended.getValue(), EPSILON);
  }

  public void testDiffStdevBehavior() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.6));
    DataModel dataModel = new GenericDataModel(users);
    Recommender recommender = new SlopeOneRecommender(dataModel);
    assertEquals(0.5, recommender.estimatePreference("test1", "2"), EPSILON);
  }

  private static Recommender buildRecommender() throws TasteException {
    DataModel dataModel = new GenericDataModel(getMockUsers());
    return new SlopeOneRecommender(dataModel);
  }

}
