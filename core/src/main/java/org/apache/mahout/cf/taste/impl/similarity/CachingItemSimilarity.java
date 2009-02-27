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
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;

import java.util.Collection;

/**
 * Caches the results from an underlying {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity} implementation.
 */
public final class CachingItemSimilarity implements ItemSimilarity {

  private final ItemSimilarity similarity;
  private final Cache<Pair<Item, Item>, Double> similarityCache;

  public CachingItemSimilarity(ItemSimilarity similarity, DataModel dataModel) throws TasteException {
    if (similarity == null) {
      throw new IllegalArgumentException("similarity is null");
    }
    this.similarity = similarity;
    int maxCacheSize = dataModel.getNumItems(); // just a dumb heuristic for sizing
    this.similarityCache = new Cache<Pair<Item, Item>, Double>(new SimilarityRetriever(similarity), maxCacheSize);
  }

  @Override
  public double itemSimilarity(Item item1, Item item2) throws TasteException {
    Pair<Item, Item> key = item1.compareTo(item2) < 0 ?
        new Pair<Item, Item>(item1, item2) :
        new Pair<Item, Item>(item2, item1);
    return similarityCache.get(key);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    similarityCache.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, similarity);
  }

  private static final class SimilarityRetriever implements Retriever<Pair<Item, Item>, Double> {
    private final ItemSimilarity similarity;

    private SimilarityRetriever(ItemSimilarity similarity) {
      this.similarity = similarity;
    }

    @Override
    public Double get(Pair<Item, Item> key) throws TasteException {
      return similarity.itemSimilarity(key.getFirst(), key.getSecond());
    }
  }

}