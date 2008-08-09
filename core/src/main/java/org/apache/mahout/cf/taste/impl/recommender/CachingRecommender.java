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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.SoftReference;
import java.util.Collections;
import java.util.List;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.Callable;

/**
 * <p>A {@link Recommender} which caches the results from another {@link Recommender} in memory.
 * Results are held by {@link SoftReference}s so that the JVM may reclaim memory from the recommendationCache
 * in low-memory situations.</p>
 */
public final class CachingRecommender implements Recommender {

  private static final Logger log = LoggerFactory.getLogger(CachingRecommender.class);

  private final Recommender recommender;
  private final AtomicInteger maxHowMany;
  private final Cache<Object, Recommendations> recommendationCache;
  private final Cache<Pair<?, ?>, Double> estimatedPrefCache;
  private final RefreshHelper refreshHelper;

  public CachingRecommender(Recommender recommender) throws TasteException {
    if (recommender == null) {
      throw new IllegalArgumentException("recommender is null");
    }
    this.recommender = recommender;
    this.maxHowMany = new AtomicInteger(1);
    // Use "num users" as an upper limit on cache size. Rough guess.
    int numUsers = recommender.getDataModel().getNumUsers();
    this.recommendationCache =
            new Cache<Object, Recommendations>(
                    new RecommendationRetriever(this.recommender, this.maxHowMany),
                    numUsers);
    this.estimatedPrefCache =
            new Cache<Pair<?, ?>, Double>(new EstimatedPrefRetriever(this.recommender), numUsers);
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      public Object call() {
        clear();
        return null;
      }
    });
    this.refreshHelper.addDependency(recommender);
  }

  public List<RecommendedItem> recommend(Object userID, int howMany) throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("user ID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    synchronized (maxHowMany) {
      if (howMany > maxHowMany.get()) {
        maxHowMany.set(howMany);
      }
    }

    Recommendations recommendations = recommendationCache.get(userID);
    if (recommendations.getItems().size() < howMany && !recommendations.noMoreRecommendableItems) {
      clear(userID);
      recommendations = recommendationCache.get(userID);
      if (recommendations.getItems().size() < howMany) {
        recommendations.noMoreRecommendableItems = true;
      }
    }

    return recommendations.getItems().size() > howMany ?
           recommendations.getItems().subList(0, howMany) :
           recommendations.getItems();
  }

  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
          throws TasteException {
    // Hmm, hard to recommendationCache this since the rescorer may change
    return recommender.recommend(userID, howMany, rescorer);
  }

  public double estimatePreference(Object userID, Object itemID) throws TasteException {
    return estimatedPrefCache.get(new Pair<Object, Object>(userID, itemID));
  }

  public void setPreference(Object userID, Object itemID, double value) throws TasteException {
    recommender.setPreference(userID, itemID, value);
    clear(userID);
  }

  public void removePreference(Object userID, Object itemID) throws TasteException {
    recommender.removePreference(userID, itemID);
    clear(userID);
  }

  public DataModel getDataModel() {
    return recommender.getDataModel();
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  /**
   * <p>Clears cached recommendations for the given user.</p>
   *
   * @param userID clear cached data associated with this user ID
   */
  public void clear(Object userID) {
    log.debug("Clearing recommendations for user ID '{}'", userID);
    recommendationCache.remove(userID);
  }

  /**
   * <p>Clears all cached recommendations.</p>
   */
  public void clear() {
    log.debug("Clearing all recommendations...");
    recommendationCache.clear();
  }

  @Override
  public String toString() {
    return "CachingRecommender[recommender:" + recommender + ']';
  }

  private static final class RecommendationRetriever implements Retriever<Object, Recommendations> {

    private final Recommender recommender;
    private final AtomicInteger maxHowMany;

    private RecommendationRetriever(Recommender recommender, AtomicInteger maxHowMany) {
      this.recommender = recommender;
      this.maxHowMany = maxHowMany;
    }

    public Recommendations get(Object key) throws TasteException {
      log.debug("Retrieving new recommendations for user ID '{}'", key);
      return new Recommendations(Collections.unmodifiableList(recommender.recommend(key, maxHowMany.get())));
    }
  }

  private static final class EstimatedPrefRetriever implements Retriever<Pair<?, ?>, Double> {

    private final Recommender recommender;

    private EstimatedPrefRetriever(Recommender recommender) {
      this.recommender = recommender;
    }

    public Double get(Pair<?, ?> key) throws TasteException {
      Object userID = key.getFirst();
      Object itemID = key.getSecond();
      log.debug("Retrieving estimated preference for user ID '{}' and item ID '{}'", userID, itemID);
      return recommender.estimatePreference(userID, itemID);
    }
  }

  private static final class Recommendations {

    private final List<RecommendedItem> items;
    private boolean noMoreRecommendableItems;

    private Recommendations(List<RecommendedItem> items) {
      this.items = items;
      this.noMoreRecommendableItems = false;
    }

    private List<RecommendedItem> getItems() {
      return items;
    }
  }

}
