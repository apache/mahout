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

import java.util.Collection;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.LongPair;

import com.google.common.base.Preconditions;

/**
 * Caches the results from an underlying {@link UserSimilarity} implementation.
 */
public final class CachingUserSimilarity implements UserSimilarity {
  
  private final UserSimilarity similarity;
  private final Cache<LongPair,Double> similarityCache;
  private final RefreshHelper refreshHelper;

  /**
   * Creates this on top of the given {@link UserSimilarity}.
   * The cache is sized according to properties of the given {@link DataModel}.
   */
  public CachingUserSimilarity(UserSimilarity similarity, DataModel dataModel) throws TasteException {
    this(similarity, dataModel.getNumUsers());
  }

  /**
   * Creates this on top of the given {@link UserSimilarity}.
   * The cache size is capped by the given size.
   */
  public CachingUserSimilarity(UserSimilarity similarity, int maxCacheSize) {
    Preconditions.checkArgument(similarity != null, "similarity is null");
    this.similarity = similarity;
    this.similarityCache = new Cache<LongPair,Double>(new SimilarityRetriever(similarity), maxCacheSize);
    this.refreshHelper = new RefreshHelper(new Callable<Void>() {
      @Override
      public Void call() {
        similarityCache.clear();
        return null;
      }
    });
    refreshHelper.addDependency(similarity);
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    LongPair key = userID1 < userID2 ? new LongPair(userID1, userID2) : new LongPair(userID2, userID1);
    return similarityCache.get(key);
  }
  
  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    similarityCache.clear();
    similarity.setPreferenceInferrer(inferrer);
  }

  public void clearCacheForUser(long userID) {
    similarityCache.removeKeysMatching(new LongPairMatchPredicate(userID));
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  private static final class SimilarityRetriever implements Retriever<LongPair,Double> {
    private final UserSimilarity similarity;
    
    private SimilarityRetriever(UserSimilarity similarity) {
      this.similarity = similarity;
    }
    
    @Override
    public Double get(LongPair key) throws TasteException {
      return similarity.userSimilarity(key.getFirst(), key.getSecond());
    }
  }
  
}
