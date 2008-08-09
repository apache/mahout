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
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;

import java.util.Collection;

/**
 * <p>Defines cluster similarity as the <em>largest</em> correlation between any two
 * {@link org.apache.mahout.cf.taste.model.User}s in the clusters -- that is, it says that clusters are close
 * when <em>some pair</em> of their members has high correlation.</p>
 */
public final class NearestNeighborClusterSimilarity implements ClusterSimilarity {

  private final UserCorrelation correlation;
  private final double samplingPercentage;

  /**
   * <p>Constructs a {@link NearestNeighborClusterSimilarity} based on the given {@link UserCorrelation}.
   * All user-user correlations are examined.</p>
   */
  public NearestNeighborClusterSimilarity(UserCorrelation correlation) {
    this(correlation, 1.0);
  }

  /**
   * <p>Constructs a {@link NearestNeighborClusterSimilarity} based on the given {@link UserCorrelation}.
   * By setting <code>samplingPercentage</code> to a value less than 1.0, this implementation will only examine
   * that fraction of all user-user correlations between two clusters, increasing performance at the expense
   * of accuracy.</p>
   */
  public NearestNeighborClusterSimilarity(UserCorrelation correlation, double samplingPercentage) {
    if (correlation == null) {
      throw new IllegalArgumentException("correlation is null");
    }
    if (Double.isNaN(samplingPercentage) || samplingPercentage <= 0.0 || samplingPercentage > 1.0) {
      throw new IllegalArgumentException("samplingPercentage is invalid: " + samplingPercentage);
    }
    this.correlation = correlation;
    this.samplingPercentage = samplingPercentage;
  }

  public double getSimilarity(Collection<User> cluster1,
                              Collection<User> cluster2) throws TasteException {
    if (cluster1.isEmpty() || cluster2.isEmpty()) {
      return Double.NaN;
    }
    double greatestCorrelation = Double.NEGATIVE_INFINITY;
    for (User user1 : cluster1) {
      if (samplingPercentage >= 1.0 || Math.random() < samplingPercentage) {
        for (User user2 : cluster2) {
          double theCorrelation = correlation.userCorrelation(user1, user2);
          if (theCorrelation > greatestCorrelation) {
            greatestCorrelation = theCorrelation;
          }
        }
      }
    }
    // We skipped everything? well, at least try comparing the first Users to get some value
    if (greatestCorrelation == Double.NEGATIVE_INFINITY) {
      return correlation.userCorrelation(cluster1.iterator().next(), cluster2.iterator().next());
    }
    return greatestCorrelation;
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, correlation);
  }

  @Override
  public String toString() {
    return "NearestNeighborClusterSimilarity[correlation:" + correlation + ']';
  }

}
