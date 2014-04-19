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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import com.google.common.collect.Lists;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItem;

import java.util.Collections;
import java.util.List;

public class TopSimilarItemsQueue extends PriorityQueue<SimilarItem> {

  private static final long SENTINEL_ID = Long.MIN_VALUE;

  private final int maxSize;

  public TopSimilarItemsQueue(int maxSize) {
    super(maxSize);
    this.maxSize = maxSize;
  }

  public List<SimilarItem> getTopItems() {
    List<SimilarItem> items = Lists.newArrayListWithCapacity(maxSize);
    while (size() > 0) {
      SimilarItem topItem = pop();
      // filter out "sentinel" objects necessary for maintaining an efficient priority queue
      if (topItem.getItemID() != SENTINEL_ID) {
        items.add(topItem);
      }
    }
    Collections.reverse(items);
    return items;
  }

  @Override
  protected boolean lessThan(SimilarItem one, SimilarItem two) {
    return one.getSimilarity() < two.getSimilarity();
  }

  @Override
  protected SimilarItem getSentinelObject() {
    return new SimilarItem(SENTINEL_ID, Double.MIN_VALUE);
  }
}
