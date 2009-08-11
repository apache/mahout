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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;

import java.util.Collection;

/** A caching wrapper around an underlying {@link UserNeighborhood} implementation. */
public final class CachingUserNeighborhood implements UserNeighborhood {

  private final UserNeighborhood neighborhood;
  private final Cache<Long, long[]> neighborhoodCache;

  public CachingUserNeighborhood(UserNeighborhood neighborhood, DataModel dataModel) throws TasteException {
    if (neighborhood == null) {
      throw new IllegalArgumentException("neighborhood is null");
    }
    this.neighborhood = neighborhood;
    int maxCacheSize = dataModel.getNumUsers(); // just a dumb heuristic for sizing
    this.neighborhoodCache = new Cache<Long, long[]>(
            new NeighborhoodRetriever(neighborhood), maxCacheSize);
  }

  @Override
  public long[] getUserNeighborhood(long userID) throws TasteException {
    return neighborhoodCache.get(userID);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    neighborhoodCache.clear();
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, neighborhood);
  }

  private static final class NeighborhoodRetriever implements Retriever<Long, long[]> {
    private final UserNeighborhood neighborhood;

    private NeighborhoodRetriever(UserNeighborhood neighborhood) {
      this.neighborhood = neighborhood;
    }

    @Override
    public long[] get(Long key) throws TasteException {
      return neighborhood.getUserNeighborhood(key);
    }
  }

}
