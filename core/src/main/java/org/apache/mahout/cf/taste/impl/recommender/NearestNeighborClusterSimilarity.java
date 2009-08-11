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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;

/**
 * <p>Defines cluster similarity as the <em>largest</em> similarity between any two users in the clusters --
 * that is, it says that clusters are close when <em>some pair</em> of their members has high similarity.</p>
 */
public final class NearestNeighborClusterSimilarity implements ClusterSimilarity {

  private final UserSimilarity similarity;
  private final double samplingRate;

  /**
   * <p>Constructs a {@link NearestNeighborClusterSimilarity} based on the given {@link UserSimilarity}. All user-user
   * similarities are examined.</p>
   */
  public NearestNeighborClusterSimilarity(UserSimilarity similarity) {
    this(similarity, 1.0);
  }

  /**
   * <p>Constructs a {@link NearestNeighborClusterSimilarity} based on the given {@link UserSimilarity}. By setting
   * <code>samplingRate</code> to a value less than 1.0, this implementation will only examine that fraction of all
   * user-user similarities between two clusters, increasing performance at the expense of accuracy.</p>
   */
  public NearestNeighborClusterSimilarity(UserSimilarity similarity, double samplingRate) {
    if (similarity == null) {
      throw new IllegalArgumentException("similarity is null");
    }
    if (Double.isNaN(samplingRate) || samplingRate <= 0.0 || samplingRate > 1.0) {
      throw new IllegalArgumentException("samplingRate is invalid: " + samplingRate);
    }
    this.similarity = similarity;
    this.samplingRate = samplingRate;
  }

  @Override
  public double getSimilarity(FastIDSet cluster1, FastIDSet cluster2) throws TasteException {
    if (cluster1.isEmpty() || cluster2.isEmpty()) {
      return Double.NaN;
    }
    LongPrimitiveIterator someUsers =
            SamplingLongPrimitiveIterator.maybeWrapIterator(cluster1.iterator(), samplingRate);
    double greatestSimilarity = Double.NEGATIVE_INFINITY;
    while (someUsers.hasNext()) {
      long userID1 = someUsers.next();
      LongPrimitiveIterator it2 = cluster2.iterator();
      while (it2.hasNext()) {
        double theSimilarity = similarity.userSimilarity(userID1, it2.next());
        if (theSimilarity > greatestSimilarity) {
          greatestSimilarity = theSimilarity;
        }
      }
    }
    // We skipped everything? well, at least try comparing the first Users to get some value
    if (greatestSimilarity == Double.NEGATIVE_INFINITY) {
      return similarity.userSimilarity(cluster1.iterator().next(), cluster2.iterator().next());
    }
    return greatestSimilarity;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, similarity);
  }

  @Override
  public String toString() {
    return "NearestNeighborClusterSimilarity[similarity:" + similarity + ']';
  }

}
