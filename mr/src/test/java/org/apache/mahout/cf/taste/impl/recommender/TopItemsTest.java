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
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveArrayIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.GenericUserSimilarity;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.List;
import java.util.Random;

/**
 * Tests for {@link TopItems}.
 */
public final class TopItemsTest extends TasteTestCase {

  @Test
  public void testTopItems() throws Exception {
    long[] ids = new long[100];
    for (int i = 0; i < 100; i++) {
      ids[i] = i;
    }
    LongPrimitiveIterator possibleItemIds = new LongPrimitiveArrayIterator(ids);
    TopItems.Estimator<Long> estimator = new TopItems.Estimator<Long>() {
      @Override
      public double estimate(Long thing) {
        return thing;
      }
    };
    List<RecommendedItem> topItems = TopItems.getTopItems(10, possibleItemIds, null, estimator);
    int gold = 99;
    for (RecommendedItem topItem : topItems) {
      assertEquals(gold, topItem.getItemID());
      assertEquals(gold--, topItem.getValue(), 0.01);
    }
  }

  @Test
  public void testTopItemsRandom() throws Exception {
    long[] ids = new long[100];
    for (int i = 0; i < 100; i++) {
      ids[i] = i;
    }
    LongPrimitiveIterator possibleItemIds = new LongPrimitiveArrayIterator(ids);
    final Random random = RandomUtils.getRandom();
    TopItems.Estimator<Long> estimator = new TopItems.Estimator<Long>() {
      @Override
      public double estimate(Long thing) {
        return random.nextDouble();
      }
    };
    List<RecommendedItem> topItems = TopItems.getTopItems(10, possibleItemIds, null, estimator);
    assertEquals(10, topItems.size());
    double last = 2.0;
    for (RecommendedItem topItem : topItems) {
      assertTrue(topItem.getValue() <= last);
      last = topItem.getItemID();
    }
  }

  @Test
  public void testTopUsers() throws Exception {
    long[] ids = new long[100];
    for (int i = 0; i < 100; i++) {
      ids[i] = i;
    }
    LongPrimitiveIterator possibleItemIds = new LongPrimitiveArrayIterator(ids);
    TopItems.Estimator<Long> estimator = new TopItems.Estimator<Long>() {
      @Override
      public double estimate(Long thing) {
        return thing;
      }
    };
    long[] topItems = TopItems.getTopUsers(10, possibleItemIds, null, estimator);
    int gold = 99;
    for (long topItem : topItems) {
      assertEquals(gold--, topItem);
    }
  }

  @Test
  public void testTopItemItem() throws Exception {
    List<GenericItemSimilarity.ItemItemSimilarity> sims = Lists.newArrayList();
    for (int i = 0; i < 99; i++) {
      sims.add(new GenericItemSimilarity.ItemItemSimilarity(i, i + 1, i / 99.0));
    }

    List<GenericItemSimilarity.ItemItemSimilarity> res = TopItems.getTopItemItemSimilarities(10, sims.iterator());
    int gold = 99;
    for (GenericItemSimilarity.ItemItemSimilarity re : res) {
      assertEquals(gold--, re.getItemID2()); //the second id should be equal to 99 to start
    }
  }

  @Test
  public void testTopItemItemAlt() throws Exception {
    List<GenericItemSimilarity.ItemItemSimilarity> sims = Lists.newArrayList();
    for (int i = 0; i < 99; i++) {
      sims.add(new GenericItemSimilarity.ItemItemSimilarity(i, i + 1, 1 - (i / 99.0)));
    }

    List<GenericItemSimilarity.ItemItemSimilarity> res = TopItems.getTopItemItemSimilarities(10, sims.iterator());
    int gold = 0;
    for (GenericItemSimilarity.ItemItemSimilarity re : res) {
      assertEquals(gold++, re.getItemID1()); //the second id should be equal to 99 to start
    }
  }

  @Test
  public void testTopUserUser() throws Exception {
    List<GenericUserSimilarity.UserUserSimilarity> sims = Lists.newArrayList();
    for (int i = 0; i < 99; i++) {
      sims.add(new GenericUserSimilarity.UserUserSimilarity(i, i + 1, i / 99.0));
    }

    List<GenericUserSimilarity.UserUserSimilarity> res = TopItems.getTopUserUserSimilarities(10, sims.iterator());
    int gold = 99;
    for (GenericUserSimilarity.UserUserSimilarity re : res) {
      assertEquals(gold--, re.getUserID2()); //the second id should be equal to 99 to start
    }
  }

  @Test
  public void testTopUserUserAlt() throws Exception {
    List<GenericUserSimilarity.UserUserSimilarity> sims = Lists.newArrayList();
    for (int i = 0; i < 99; i++) {
      sims.add(new GenericUserSimilarity.UserUserSimilarity(i, i + 1, 1 - (i / 99.0)));
    }

    List<GenericUserSimilarity.UserUserSimilarity> res = TopItems.getTopUserUserSimilarities(10, sims.iterator());
    int gold = 0;
    for (GenericUserSimilarity.UserUserSimilarity re : res) {
      assertEquals(gold++, re.getUserID1()); //the first id should be equal to 0 to start
    }
  }

}
