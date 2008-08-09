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

import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.impl.correlation.PearsonCorrelation;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * <p>Tests {@link GenericUserBasedRecommender}.</p>
 */
public final class GenericUserBasedRecommenderTest extends TasteTestCase {

  public void testRecommender() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.3, firstRecommended.getValue());
    recommender.refresh(null);
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.3, firstRecommended.getValue());
  }

  public void testHowMany() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.6));
    users.add(getUser("test3", 0.4, 0.4, 0.5, 0.9));
    users.add(getUser("test4", 0.1, 0.4, 0.5, 0.8, 0.9, 1.0));
    users.add(getUser("test5", 0.2, 0.3, 0.6, 0.7, 0.1, 0.2));
    DataModel dataModel = new GenericDataModel(users);
    UserCorrelation correlation = new PearsonCorrelation(dataModel);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, correlation, dataModel);
    Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, correlation);
    List<RecommendedItem> fewRecommended = recommender.recommend("test1", 2);
    List<RecommendedItem> moreRecommended = recommender.recommend("test1", 4);
    for (int i = 0; i < fewRecommended.size(); i++) {
      assertEquals(fewRecommended.get(i).getItem(), moreRecommended.get(i).getItem());
    }
    recommender.refresh(null);
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
    UserCorrelation correlation = new PearsonCorrelation(dataModel);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(1, correlation, dataModel);
    Recommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, correlation);
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
    assertEquals(0.3, recommender.estimatePreference("test1", "2"));
  }

  public void testBestRating() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    // item one should be recommended because it has a greater rating/score
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.3, firstRecommended.getValue(), EPSILON);
  }

  public void testMostSimilar() throws Exception {
    UserBasedRecommender recommender = buildRecommender();
    List<User> similar = recommender.mostSimilarUsers("test1", 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    User first = similar.get(0);
    User second = similar.get(1);
    assertEquals("test2", first.getID());
    assertEquals("test3", second.getID());
  }

  public void testIsolatedUser() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.6));
    users.add(getUser("test3", 0.4, 0.4, 0.5, 0.9));
    users.add(getUser("test4"));
    DataModel dataModel = new GenericDataModel(users);
    UserCorrelation correlation = new PearsonCorrelation(dataModel);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(3, correlation, dataModel);
    UserBasedRecommender recommender = new GenericUserBasedRecommender(dataModel, neighborhood, correlation);
    Collection<User> mostSimilar = recommender.mostSimilarUsers("test4", 3);
    assertNotNull(mostSimilar);
    assertEquals(0, mostSimilar.size());
  }

  private static UserBasedRecommender buildRecommender() throws Exception {
    DataModel dataModel = new GenericDataModel(getMockUsers());
    UserCorrelation correlation = new PearsonCorrelation(dataModel);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(1, correlation, dataModel);
    return new GenericUserBasedRecommender(dataModel, neighborhood, correlation);
  }

}
