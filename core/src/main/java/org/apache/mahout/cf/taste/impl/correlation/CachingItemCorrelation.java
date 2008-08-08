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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Retriever;

/**
 * Caches the results from an underlying {@link ItemCorrelation} implementation.
 */
public final class CachingItemCorrelation implements ItemCorrelation {

  private final ItemCorrelation correlation;
  private final Cache<Pair<Item, Item>, Double> correlationCache;

  public CachingItemCorrelation(ItemCorrelation correlation) {
    if (correlation == null) {
      throw new IllegalArgumentException("correlation is null");
    }
    this.correlation = correlation;
    this.correlationCache = new Cache<Pair<Item, Item>, Double>(new CorrelationRetriever(correlation));
  }

  public double itemCorrelation(Item item1, Item item2) throws TasteException {
    Pair<Item, Item> key;
    if (item1.compareTo(item2) < 0) {
      key = new Pair<Item, Item>(item1, item2);
    } else {
      key = new Pair<Item, Item>(item2, item1);
    }
    return correlationCache.get(key);
  }

  public void refresh() {
    correlationCache.clear();
    correlation.refresh();
  }

  private static final class CorrelationRetriever implements Retriever<Pair<Item, Item>, Double> {
    private final ItemCorrelation correlation;
    private CorrelationRetriever(ItemCorrelation correlation) {
      this.correlation = correlation;
    }
    public Double get(Pair<Item, Item> key) throws TasteException {
      return correlation.itemCorrelation(key.getFirst(), key.getSecond());
    }
  }

}