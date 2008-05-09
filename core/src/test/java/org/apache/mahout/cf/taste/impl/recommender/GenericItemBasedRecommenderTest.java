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

import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.impl.correlation.GenericItemCorrelation;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * <p>Tests {@link GenericItemBasedRecommender}.</p>
 */
public final class GenericItemBasedRecommenderTest extends TasteTestCase {

  public void testRecommender() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.1, firstRecommended.getValue(), EPSILON);
  }

  public void testHowMany() throws Exception {
    List<User> users = new ArrayList<User>(3);
    users.add(getUser("test1", 0.1, 0.2));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.6));
    users.add(getUser("test3", 0.4, 0.4, 0.5, 0.9));
    users.add(getUser("test4", 0.1, 0.4, 0.5, 0.8, 0.9, 1.0));
    users.add(getUser("test5", 0.2, 0.3, 0.6, 0.7, 0.1, 0.2));
    DataModel dataModel = new GenericDataModel(users);
    Collection<GenericItemCorrelation.ItemItemCorrelation> correlations =
            new ArrayList<GenericItemCorrelation.ItemItemCorrelation>(6);
    for (int i = 0; i < 6; i++) {
      for (int j = i + 1; j < 6; j++) {
        correlations.add(
                new GenericItemCorrelation.ItemItemCorrelation(new GenericItem<String>(String.valueOf(i)),
                                                               new GenericItem<String>(String.valueOf(j)),
                                                               1.0 / (1.0 + (double) i + (double) j)));
      }
    }
    ItemCorrelation correlation = new GenericItemCorrelation(correlations);
    Recommender recommender = new GenericItemBasedRecommender(dataModel, correlation);
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
    Item item1 = new GenericItem<String>("0");
    Item item2 = new GenericItem<String>("1");
    Item item3 = new GenericItem<String>("2");
    Item item4 = new GenericItem<String>("3");
    Collection<GenericItemCorrelation.ItemItemCorrelation> correlations =
            new ArrayList<GenericItemCorrelation.ItemItemCorrelation>(6);
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item2, 1.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item3, 0.5));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item4, 0.2));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item2, item3, 0.7));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item2, item4, 0.5));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item3, item4, 0.9));
    ItemCorrelation correlation = new GenericItemCorrelation(correlations);
    Recommender recommender = new GenericItemBasedRecommender(dataModel, correlation);
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
    assertEquals(0.1, recommender.estimatePreference("test1", "2"), EPSILON);
  }

  /**
   * Contributed test case that verifies fix for bug
   * <a href="http://sourceforge.net/tracker/index.php?func=detail&amp;aid=1396128&amp;group_id=138771&amp;atid=741665">
   * 1396128</a>.
   */
  public void testBestRating() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend("test1", 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    // item one should be recommended because it has a greater rating/score
    assertEquals(new GenericItem<String>("2"), firstRecommended.getItem());
    assertEquals(0.1, firstRecommended.getValue(), EPSILON);
  }

  public void testMostSimilar() throws Exception {
    ItemBasedRecommender recommender = buildRecommender();
    List<RecommendedItem> similar = recommender.mostSimilarItems("0", 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals("1", first.getItem().getID());
    assertEquals(1.0, first.getValue(), EPSILON);
    assertEquals("2", second.getItem().getID());
    assertEquals(0.5, second.getValue(), EPSILON);
  }

  public void testMostSimilarToMultiple() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<Object> itemIDs = new ArrayList<Object>(2);
    itemIDs.add("0");
    itemIDs.add("1");
    List<RecommendedItem> similar = recommender.mostSimilarItems(itemIDs, 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals("2", first.getItem().getID());
    assertEquals(0.85, first.getValue(), EPSILON);
    assertEquals("3", second.getItem().getID());
    assertEquals(-0.3, second.getValue(), EPSILON);
  }

  public void testRecommendedBecause() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> recommendedBecause = recommender.recommendedBecause("test1", "4", 3);
    assertNotNull(recommendedBecause);
    assertEquals(3, recommendedBecause.size());
    RecommendedItem first = recommendedBecause.get(0);
    RecommendedItem second = recommendedBecause.get(1);
    RecommendedItem third = recommendedBecause.get(2);
    assertEquals("2", first.getItem().getID());
    assertEquals(0.99, first.getValue(), EPSILON);
    assertEquals("3", second.getItem().getID());
    assertEquals(0.4, second.getValue(), EPSILON);
    assertEquals("0", third.getItem().getID());
    assertEquals(0.2, third.getValue(), EPSILON);
  }

  private static ItemBasedRecommender buildRecommender() {
    DataModel dataModel = new GenericDataModel(getMockUsers());
    Collection<GenericItemCorrelation.ItemItemCorrelation> correlations =
            new ArrayList<GenericItemCorrelation.ItemItemCorrelation>(2);
    Item item1 = new GenericItem<String>("0");
    Item item2 = new GenericItem<String>("1");
    Item item3 = new GenericItem<String>("2");
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item2, 1.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item3, 0.5));
    ItemCorrelation correlation = new GenericItemCorrelation(correlations);
    return new GenericItemBasedRecommender(dataModel, correlation);
  }

  private static ItemBasedRecommender buildRecommender2() {
    List<User> users = new ArrayList<User>(4);
    users.add(getUser("test1", 0.1, 0.3, 0.9, 0.8));
    users.add(getUser("test2", 0.2, 0.3, 0.3, 0.4));
    users.add(getUser("test3", 0.4, 0.3, 0.5, 0.1, 0.1));
    users.add(getUser("test4", 0.7, 0.3, 0.8, 0.5, 0.6));
    DataModel dataModel = new GenericDataModel(users);
    Collection<GenericItemCorrelation.ItemItemCorrelation> correlations =
            new ArrayList<GenericItemCorrelation.ItemItemCorrelation>(10);
    Item item1 = new GenericItem<String>("0");
    Item item2 = new GenericItem<String>("1");
    Item item3 = new GenericItem<String>("2");
    Item item4 = new GenericItem<String>("3");
    Item item5 = new GenericItem<String>("4");
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item2, 1.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item3, 0.8));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item4, -0.6));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item1, item5, 1.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item2, item3, 0.9));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item2, item4, 0.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item2, item2, 1.0));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item3, item4, -0.1));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item3, item5, 0.1));
    correlations.add(new GenericItemCorrelation.ItemItemCorrelation(item4, item5, -0.5));
    ItemCorrelation correlation = new GenericItemCorrelation(correlations);
    return new GenericItemBasedRecommender(dataModel, correlation);
  }

}
