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

import com.google.common.collect.Lists;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import java.util.Collections;
import java.util.List;

public class TopItemsQueue extends PriorityQueue<MutableRecommendedItem> {

  private static final long SENTINEL_ID = Long.MIN_VALUE;

  private final int maxSize;

  public TopItemsQueue(int maxSize) {
    super(maxSize);
    this.maxSize = maxSize;
  }

  public List<RecommendedItem> getTopItems() {
    List<RecommendedItem> recommendedItems = Lists.newArrayListWithCapacity(maxSize);
    while (size() > 0) {
      MutableRecommendedItem topItem = pop();
      // filter out "sentinel" objects necessary for maintaining an efficient priority queue
      if (topItem.getItemID() != SENTINEL_ID) {
        recommendedItems.add(topItem);
      }
    }
    Collections.reverse(recommendedItems);
    return recommendedItems;
  }

  @Override
  protected boolean lessThan(MutableRecommendedItem one, MutableRecommendedItem two) {
    return one.getValue() < two.getValue();
  }

  @Override
  protected MutableRecommendedItem getSentinelObject() {
    return new MutableRecommendedItem(SENTINEL_ID, Float.MIN_VALUE);
  }
}
