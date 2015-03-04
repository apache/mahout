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

package org.apache.mahout.cf.taste.similarity.precompute;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class SimilarItemsTest extends TasteTestCase {

    @Test
    public void testIterator() {

        List<RecommendedItem> recommendedItems = new ArrayList<RecommendedItem>();
        for (long itemId = 2; itemId < 10; itemId++) {
            recommendedItems.add(new GenericRecommendedItem(itemId, itemId));
        }

        SimilarItems similarItems = new SimilarItems(1,recommendedItems);

        Iterator<SimilarItem> itemIterator = similarItems.getSimilarItems().iterator();
        int byHandIndex = 0;
        for (SimilarItem simItem = itemIterator.next(); itemIterator.hasNext(); simItem = itemIterator.next()) {
            RecommendedItem recItem = recommendedItems.get(byHandIndex++);
            assertEquals(simItem.getItemID(), recItem.getItemID());
            assertEquals(simItem.getSimilarity(), recItem.getValue(), EPSILON);
        }

    }
}
