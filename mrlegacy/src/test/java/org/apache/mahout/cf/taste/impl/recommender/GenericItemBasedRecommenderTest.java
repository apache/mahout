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
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.MostSimilarItemsCandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.easymock.EasyMock;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/** <p>Tests {@link GenericItemBasedRecommender}.</p> */
public final class GenericItemBasedRecommenderTest extends TasteTestCase {

  @Test
  public void testRecommender() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend(1, 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1f, firstRecommended.getValue(), EPSILON);
    recommender.refresh(null);
    recommended = recommender.recommend(1, 1);
    firstRecommended = recommended.get(0);    
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1f, firstRecommended.getValue(), EPSILON);
  }

  @Test
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

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities = Lists.newArrayList();
    for (int i = 0; i < 6; i++) {
      for (int j = i + 1; j < 6; j++) {
        similarities.add(
            new GenericItemSimilarity.ItemItemSimilarity(i, j, 1.0 / (1.0 + i + j)));
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

  @Test
  public void testRescorer() throws Exception {

    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {0.1, 0.2},
                    {0.2, 0.3, 0.3, 0.6},
                    {0.4, 0.4, 0.5, 0.9},
            });

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities = Lists.newArrayList();
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

  @Test
  public void testEstimatePref() throws Exception {
    Recommender recommender = buildRecommender();
    assertEquals(0.1f, recommender.estimatePreference(1, 2), EPSILON);
  }

  /**
   * Contributed test case that verifies fix for bug
   * <a href="http://sourceforge.net/tracker/index.php?func=detail&amp;aid=1396128&amp;group_id=138771&amp;atid=741665">
   * 1396128</a>.
   */
  @Test
  public void testBestRating() throws Exception {
    Recommender recommender = buildRecommender();
    List<RecommendedItem> recommended = recommender.recommend(1, 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    // item one should be recommended because it has a greater rating/score
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.1f, firstRecommended.getValue(), EPSILON);
  }

  @Test
  public void testMostSimilar() throws Exception {
    ItemBasedRecommender recommender = buildRecommender();
    List<RecommendedItem> similar = recommender.mostSimilarItems(0, 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals(1, first.getItemID());
    assertEquals(1.0f, first.getValue(), EPSILON);
    assertEquals(2, second.getItemID());
    assertEquals(0.5f, second.getValue(), EPSILON);
  }

  @Test
  public void testMostSimilarToMultiple() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> similar = recommender.mostSimilarItems(new long[] {0, 1}, 2);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals(2, first.getItemID());
    assertEquals(0.85f, first.getValue(), EPSILON);
    assertEquals(3, second.getItemID());
    assertEquals(-0.3f, second.getValue(), EPSILON);
  }

  @Test
  public void testMostSimilarToMultipleExcludeIfNotSimilarToAll() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> similar = recommender.mostSimilarItems(new long[] {3, 4}, 2);
    assertNotNull(similar);
    assertEquals(1, similar.size());
    RecommendedItem first = similar.get(0);
    assertEquals(0, first.getItemID());
    assertEquals(0.2f, first.getValue(), EPSILON);
  }

  @Test
  public void testMostSimilarToMultipleDontExcludeIfNotSimilarToAll() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> similar = recommender.mostSimilarItems(new long[] {1, 2, 4}, 10, false);
    assertNotNull(similar);
    assertEquals(2, similar.size());
    RecommendedItem first = similar.get(0);
    RecommendedItem second = similar.get(1);
    assertEquals(0, first.getItemID());
    assertEquals(0.933333333f, first.getValue(), EPSILON);
    assertEquals(3, second.getItemID());
    assertEquals(-0.2f, second.getValue(), EPSILON);
  }


  @Test
  public void testRecommendedBecause() throws Exception {
    ItemBasedRecommender recommender = buildRecommender2();
    List<RecommendedItem> recommendedBecause = recommender.recommendedBecause(1, 4, 3);
    assertNotNull(recommendedBecause);
    assertEquals(3, recommendedBecause.size());
    RecommendedItem first = recommendedBecause.get(0);
    RecommendedItem second = recommendedBecause.get(1);
    RecommendedItem third = recommendedBecause.get(2);
    assertEquals(2, first.getItemID());
    assertEquals(0.99f, first.getValue(), EPSILON);
    assertEquals(3, second.getItemID());
    assertEquals(0.4f, second.getValue(), EPSILON);
    assertEquals(0, third.getItemID());
    assertEquals(0.2f, third.getValue(), EPSILON);
  }

  private static ItemBasedRecommender buildRecommender() {
    DataModel dataModel = getDataModel();
    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities = Lists.newArrayList();
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 1, 1.0));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(0, 2, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 2, 0.0));
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

    Collection<GenericItemSimilarity.ItemItemSimilarity> similarities = Lists.newArrayList();
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


  /**
   * we're making sure that a user's preferences are fetched only once from the {@link DataModel} for one call to
   * {@link GenericItemBasedRecommender#recommend(long, int)}
   *
   * @throws Exception
   */
  @Test
  public void preferencesFetchedOnlyOnce() throws Exception {

    DataModel dataModel = EasyMock.createMock(DataModel.class);
    ItemSimilarity itemSimilarity = EasyMock.createMock(ItemSimilarity.class);
    CandidateItemsStrategy candidateItemsStrategy = EasyMock.createMock(CandidateItemsStrategy.class);
    MostSimilarItemsCandidateItemsStrategy mostSimilarItemsCandidateItemsStrategy =
        EasyMock.createMock(MostSimilarItemsCandidateItemsStrategy.class);

    PreferenceArray preferencesFromUser = new GenericUserPreferenceArray(
        Arrays.asList(new GenericPreference(1L, 1L, 5.0f), new GenericPreference(1L, 2L, 4.0f)));

    EasyMock.expect(dataModel.getMinPreference()).andReturn(Float.NaN);
    EasyMock.expect(dataModel.getMaxPreference()).andReturn(Float.NaN);

    EasyMock.expect(dataModel.getPreferencesFromUser(1L)).andReturn(preferencesFromUser);
    EasyMock.expect(candidateItemsStrategy.getCandidateItems(1L, preferencesFromUser, dataModel))
        .andReturn(new FastIDSet(new long[] { 3L, 4L }));

    EasyMock.expect(itemSimilarity.itemSimilarities(3L, preferencesFromUser.getIDs()))
        .andReturn(new double[] { 0.5, 0.3 });
    EasyMock.expect(itemSimilarity.itemSimilarities(4L, preferencesFromUser.getIDs()))
        .andReturn(new double[] { 0.4, 0.1 });

    EasyMock.replay(dataModel, itemSimilarity, candidateItemsStrategy, mostSimilarItemsCandidateItemsStrategy);

    Recommender recommender = new GenericItemBasedRecommender(dataModel, itemSimilarity,
        candidateItemsStrategy, mostSimilarItemsCandidateItemsStrategy);

    recommender.recommend(1L, 3);

    EasyMock.verify(dataModel, itemSimilarity, candidateItemsStrategy, mostSimilarItemsCandidateItemsStrategy);
  }
}
