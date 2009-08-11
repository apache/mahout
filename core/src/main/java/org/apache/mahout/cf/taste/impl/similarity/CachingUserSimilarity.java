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
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;

/** Caches the results from an underlying {@link org.apache.mahout.cf.taste.similarity.UserSimilarity} implementation. */
public final class CachingUserSimilarity implements UserSimilarity {

  private final UserSimilarity similarity;
  private final Cache<LongPair, Double> similarityCache;

  public CachingUserSimilarity(UserSimilarity similarity, DataModel dataModel) throws TasteException {
    if (similarity == null) {
      throw new IllegalArgumentException("similarity is null");
    }
    this.similarity = similarity;
    int maxCacheSize = dataModel.getNumUsers(); // just a dumb heuristic for sizing    
    this.similarityCache = new Cache<LongPair, Double>(new SimilarityRetriever(similarity), maxCacheSize);
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

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    similarityCache.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, similarity);
  }

  private static final class SimilarityRetriever implements Retriever<LongPair, Double> {
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
