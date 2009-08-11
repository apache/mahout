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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** <p>Tests {@link GenericItemBasedRecommender}.</p> */
public final class GenericItemBasedRecommenderTest extends TasteTestCase {

  public void testRecommender() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend(1, 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1, firstRecommended.getValue(), EPSILON);
    recommender.refresh(null);
    recommended = recommender.recommend(1, 1);
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1, firstRecommended.getValue(), EPSILON);
  }

  public void testHowMany() throws Exception {

    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3, 4, 5},
            new Double[][] {
                    {0.1, 0.2},
                    {0.2, 0.3, 0.3, 0.6},
                    {0.4, 0.4, 0.5, 0.9},
                    {0.1, 0.4, 0.5, 0.8, 0.9, 1.0},
                    {0.2, 0.3, 0.6, 0.7, 0.1, 0.2},
            });

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities =
        new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(6);
    for (int i = 0; i < 6; i++) {
      for (int j = i + 1; j < 6; j++) {
        similarities.add(
            new GenericItemSimilarity.ItemItemSimilarity(i, j, 1.0 / (1.0 + (double) i + (double) j)));
      }
    }
    ItemSimilarity similarity = new GenericItemSimilarity(similarities);
    Recommender recommender = new GenericItemBasedRecommender(dataModel, similarity);
    List<RecommendedItem> fewRecommended = recommender.recommend(1, 2);
    List<RecommendedItem> moreRecommended = recommender.recommend(1, 4);
    for (int i = 0; i < fewRecommended.size(); i++) {
      assertEquals(fewRecommended.get(i).getItemID(), moreRecommended.get(i).getItemID());
    }
    recommender.refresh(null);
    for (int i = 0; i < fewRecommended.size(); i++) {
      assertEquals(fewRecommended.get(i).getItemID(), moreRecommended.get(i).getItemID());
    }
  }

  public void testRescorer() throws Exception {

    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {0.1, 0.2},
                    {0.2, 0.3, 0.3, 0.6},
                    {0.4, 0.4, 0.5, 0.9},
            });

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities =
        new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(6);
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 1, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 2, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 3, 0.2));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 2, 0.7));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 3, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(2, 3, 0.9));
    ItemSimilarity similarity = new GenericItemSimilarity(similarities);
    Recommender recommender = new GenericItemBasedRecommender(dataModel, similarity);
    List<RecommendedItem> originalRecommended = recommender.recommend(1, 2);
    List<RecommendedItem> rescoredRecommended =
        recommender.recommend(1, 2, new ReversingRescorer<Long>());
    assertNotNull(originalRecommended);
    assertNotNull(rescoredRecommended);
    assertEquals(2, originalRecommended.size());
    assertEquals(2, rescoredRecommended.size());
    assertEquals(originalRecommended.get(0).getItemID(), rescoredRecommended.get(1).getItemID());
    assertEquals(originalRecommended.get(1).getItemID(), rescoredRecommended.get(0).getItemID());
  }

  public void testEstimatePref() throws Exception {
    Recommender recommender = buildRecommender();
    assertEquals(0.1, recommender.estimatePreference(1, 2), EPSILON);
  }

  /**
   * Contributed test case that verifies fix for bug <a href="http://sourceforge.net/tracker/index.php?func=detail&amp;aid=1396128&amp;group_id=138771&amp;atid=741665">
   * 1396128</a>.
   */
  public void testBestRating() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend(1, 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    // item one should be recommended because it has a greater rating/score
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1, firstRecommended.getValue(), EPSILON);
  }

  public void testMostSimilar() throws Exception {
    ItemBasedRecommender recommender = buildRecommender();
    List<RecommendedItem> similar = recommender.mostSimilarItems(0, 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals(1, first.getItemID());
    assertEquals(1.0, first.getValue(), EPSILON);
    assertEquals(2, second.getItemID());
    assertEquals(0.5, second.getValue(), EPSILON);
  }

  public void testMostSimilarToMultiple() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> similar = recommender.mostSimilarItems(new long[] {0, 1}, 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals(2, first.getItemID());
    assertEquals(0.85, first.getValue(), EPSILON);
    assertEquals(3, second.getItemID());
    assertEquals(-0.3, second.getValue(), EPSILON);
  }

  public void testRecommendedBecause() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> recommendedBecause = recommender.recommendedBecause(1, 4, 3);
    assertNotNull(recommendedBecause);
    assertEquals(3, recommendedBecause.size());
    RecommendedItem first = recommendedBecause.get(0);
    RecommendedItem second = recommendedBecause.get(1);
    RecommendedItem third = recommendedBecause.get(2);
    assertEquals(2, first.getItemID());
    assertEquals(0.99, first.getValue(), EPSILON);
    assertEquals(3, second.getItemID());
    assertEquals(0.4, second.getValue(), EPSILON);
    assertEquals(0, third.getItemID());
    assertEquals(0.2, third.getValue(), EPSILON);
  }

  private static ItemBasedRecommender buildRecommender() {
    DataModel dataModel = getDataModel();
    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities =
        new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(2);
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 1, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 2, 0.5));
    ItemSimilarity similarity = new GenericItemSimilarity(similarities);
    return new GenericItemBasedRecommender(dataModel, similarity);
  }

  private static ItemBasedRecommender buildRecommender2() {

    DataModel dataModel = getDataModel(
        new long[] {1, 2, 3, 4},
        new Double[][] {
                {0.1, 0.3, 0.9, 0.8},
                {0.2, 0.3, 0.3, 0.4},
                {0.4, 0.3, 0.5, 0.1, 0.1},
                {0.7, 0.3, 0.8, 0.5, 0.6},
        });

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities =
        new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(10);
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 1, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 2, 0.8));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 3, -0.6));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 4, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 2, 0.9));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 3, 0.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 1, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(2, 3, -0.1));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(2, 4, 0.1));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(3, 4, -0.5));
    ItemSimilarity similarity = new GenericItemSimilarity(similarities);
    return new GenericItemBasedRecommender(dataModel, similarity);
  }

}
