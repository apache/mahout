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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.util.List;

public class TopItemsQueueTest extends TasteTestCase {

  @Test
  public void topK() {

    float[] ratings = {0.5f, 0.6f, 0.7f, 2.0f, 0.0f};

    List<RecommendedItem> topItems = findTop(ratings, 2);

    assertEquals(2, topItems.size());
    assertEquals(3L, topItems.get(0).getItemID());
    assertEquals(2.0f, topItems.get(0).getValue(), MahoutTestCase.EPSILON);
    assertEquals(2L, topItems.get(1).getItemID());
    assertEquals(0.7f, topItems.get(1).getValue(), MahoutTestCase.EPSILON);
  }

  @Test
  public void topKInputSmallerThanK() {

    float[] ratings = {0.7f, 2.0f};

    List<RecommendedItem> topItems = findTop(ratings, 3);

    assertEquals(2, topItems.size());
    assertEquals(1L, topItems.get(0).getItemID());
    assertEquals(2.0f, topItems.get(0).getValue(), MahoutTestCase.EPSILON);
    assertEquals(0L, topItems.get(1).getItemID());
    assertEquals(0.7f, topItems.get(1).getValue(), MahoutTestCase.EPSILON);
  }


  private static List<RecommendedItem> findTop(float[] ratings, int k) {
    TopItemsQueue queue = new TopItemsQueue(k);

    for (int item = 0; item < ratings.length; item++) {
      MutableRecommendedItem top = queue.top();
      if (ratings[item] > top.getValue()) {
        top.set(item, ratings[item]);
        queue.updateTop();
      }
    }

    return queue.getTopItems();
  }

}
