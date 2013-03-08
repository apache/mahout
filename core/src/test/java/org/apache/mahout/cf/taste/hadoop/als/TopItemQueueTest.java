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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.junit.Test;

import java.util.List;

public class TopItemQueueTest extends TasteTestCase {

  @Test
  public void topK() {

    float[] ratings = { .5f, .6f, .7f, 2f, 0f };

    List<RecommendedItem> topItems = findTop(ratings, 2);

    assertEquals(2, topItems.size());
    assertEquals(3l, topItems.get(0).getItemID());
    assertEquals(2f, topItems.get(0).getValue(), TasteTestCase.EPSILON);
    assertEquals(2l, topItems.get(1).getItemID());
    assertEquals(.7f, topItems.get(1).getValue(), TasteTestCase.EPSILON);
  }

  @Test
  public void topKInputSmallerThanK() {

    float[] ratings = {.7f, 2f};

    List<RecommendedItem> topItems = findTop(ratings, 3);

    assertEquals(2, topItems.size());
    assertEquals(1l, topItems.get(0).getItemID());
    assertEquals(2f, topItems.get(0).getValue(), TasteTestCase.EPSILON);
    assertEquals(0l, topItems.get(1).getItemID());
    assertEquals(.7f, topItems.get(1).getValue(), TasteTestCase.EPSILON);
  }


  private List<RecommendedItem> findTop(float[] ratings, int k) {
    TopItemQueue queue = new TopItemQueue(k);

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
