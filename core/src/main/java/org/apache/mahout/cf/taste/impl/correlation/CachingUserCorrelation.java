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

import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.correlation.PreferenceInferrer;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;

import java.util.Collection;

/**
 * Caches the results from an underlying {@link UserCorrelation} implementation.
 */
public final class CachingUserCorrelation implements UserCorrelation {

  private final UserCorrelation correlation;
  private final Cache<Pair<User, User>, Double> correlationCache;

  public CachingUserCorrelation(UserCorrelation correlation, DataModel dataModel) throws TasteException {
    if (correlation == null) {
      throw new IllegalArgumentException("correlation is null");
    }
    this.correlation = correlation;
    int maxCacheSize = dataModel.getNumUsers(); // just a dumb heuristic for sizing    
    this.correlationCache = new Cache<Pair<User, User>, Double>(new CorrelationRetriever(correlation), maxCacheSize);
  }

  public double userCorrelation(User user1, User user2) throws TasteException {
    Pair<User, User> key;
    if (user1.compareTo(user2) < 0) {
      key = new Pair<User, User>(user1, user2);
    } else {
      key = new Pair<User, User>(user2, user1);
    }
    return correlationCache.get(key);
  }

  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    correlationCache.clear();
    correlation.setPreferenceInferrer(inferrer);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    correlationCache.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, correlation);
  }

  private static final class CorrelationRetriever implements Retriever<Pair<User, User>, Double> {
    private final UserCorrelation correlation;
    private CorrelationRetriever(UserCorrelation correlation) {
      this.correlation = correlation;
    }
    public Double get(Pair<User, User> key) throws TasteException {
      return correlation.userCorrelation(key.getFirst(), key.getSecond());
    }
  }

}
