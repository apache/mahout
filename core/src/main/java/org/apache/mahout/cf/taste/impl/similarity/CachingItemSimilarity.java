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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.LongPair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.util.Collection;

/** Caches the results from an underlying {@link ItemSimilarity} implementation. */
public final class CachingItemSimilarity implements ItemSimilarity {

  private final ItemSimilarity similarity;
  private final Cache<LongPair, Double> similarityCache;

  public CachingItemSimilarity(ItemSimilarity similarity, DataModel dataModel) throws TasteException {
    if (similarity == null) {
      throw new IllegalArgumentException("similarity is null");
    }
    this.similarity = similarity;
    int maxCacheSize = dataModel.getNumItems(); // just a dumb heuristic for sizing
    this.similarityCache = new Cache<LongPair, Double>(new SimilarityRetriever(similarity), maxCacheSize);
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    LongPair key = itemID1 < itemID2 ? new LongPair(itemID1, itemID2) : new LongPair(itemID2, itemID1);
    return similarityCache.get(key);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    similarityCache.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, similarity);
  }

  private static final class SimilarityRetriever implements Retriever<LongPair, Double> {
    private final ItemSimilarity similarity;

    private SimilarityRetriever(ItemSimilarity similarity) {
      this.similarity = similarity;
    }

    @Override
    public Double get(LongPair key) throws TasteException {
      return similarity.itemSimilarity(key.getFirst(), key.getSecond());
    }
  }

}