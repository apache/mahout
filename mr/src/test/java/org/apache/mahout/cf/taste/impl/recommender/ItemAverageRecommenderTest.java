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
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.junit.Test;

import java.util.List;

public final class ItemAverageRecommenderTest extends TasteTestCase {

  @Test
  public void testRecommender() throws Exception {
    Recommender recommender = new ItemAverageRecommender(getDataModel());
    List<RecommendedItem> recommended = recommender.recommend(1, 1);
    assertNotNull(recommended);
    assertEquals(1, recommended.size());
    RecommendedItem firstRecommended = recommended.get(0);
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.53333336f, firstRecommended.getValue(), EPSILON);
    recommender.refresh(null);
    assertEquals(2, firstRecommended.getItemID());
    assertEquals(0.53333336f, firstRecommended.getValue(), EPSILON);
  }

}