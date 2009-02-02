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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * <p>Computes a neighborhood consisting of the nearest n {@link User}s to a given {@link User}.
 * "Nearest" is defined by the given {@link org.apache.mahout.cf.taste.similarity.UserSimilarity}.</p>
 */
public final class NearestNUserNeighborhood extends AbstractUserNeighborhood {

  private static final Logger log = LoggerFactory.getLogger(NearestNUserNeighborhood.class);

  private final int n;
  private final double minSimilarity;

  /**
   * @param n neighborhood size
   * @param userSimilarity nearness metric
   * @param dataModel data model
   * @throws IllegalArgumentException if n &lt; 1, or userSimilarity or dataModel are <code>null</code>
   */
  public NearestNUserNeighborhood(int n,
                                  UserSimilarity userSimilarity,
                                  DataModel dataModel) {
    this(n, Double.NEGATIVE_INFINITY, userSimilarity, dataModel, 1.0);
  }

  /**
   * @param n neighborhood size
   * @param minSimilarity minimal similarity required for neighbors
   * @param userSimilarity nearness metric
   * @param dataModel data model
   * @throws IllegalArgumentException if n &lt; 1, or userSimilarity or dataModel are <code>null</code>
   */
  public NearestNUserNeighborhood(int n, double minSimilarity,
                                  UserSimilarity userSimilarity,
                                  DataModel dataModel) {
    this(n, minSimilarity, userSimilarity, dataModel, 1.0);
  }

  /**
   * @param n neighborhood size
   * @param minSimilarity minimal similarity required for neighbors
   * @param userSimilarity nearness metric
   * @param dataModel data model
   * @param samplingRate percentage of users to consider when building neighborhood -- decrease to
   * trade quality for performance
   * @throws IllegalArgumentException if n &lt; 1 or samplingRate is NaN or not in (0,1],
   * or userSimilarity or dataModel are <code>null</code>
   */
  public NearestNUserNeighborhood(int n, double minSimilarity,
                                  UserSimilarity userSimilarity,
                                  DataModel dataModel,
                                  double samplingRate) {
    super(userSimilarity, dataModel, samplingRate);
    if (n < 1) {
      throw new IllegalArgumentException("n must be at least 1");
    }
    this.n = n;
    this.minSimilarity = minSimilarity;
  }

  @Override
  public Collection<User> getUserNeighborhood(Object userID) throws TasteException {
    log.trace("Computing neighborhood around user ID '{}'", userID);

    DataModel dataModel = getDataModel();
    User theUser = dataModel.getUser(userID);
    UserSimilarity userSimilarityImpl = getUserSimilarity();

    TopItems.Estimator<User> estimator = new Estimator(userSimilarityImpl, theUser, minSimilarity);

    List<User> neighborhood = TopItems.getTopUsers(n, dataModel.getUsers(), null, estimator);

    log.trace("UserNeighborhood around user ID '{}' is: {}", userID, neighborhood);

    return Collections.unmodifiableList(neighborhood);
  }

  @Override
  public String toString() {
    return "NearestNUserNeighborhood";
  }

  private static class Estimator implements TopItems.Estimator<User> {
    private final UserSimilarity userSimilarityImpl;
    private final User theUser;
    private final double minSim;

    private Estimator(UserSimilarity userSimilarityImpl, User theUser, double minSim) {
      this.userSimilarityImpl = userSimilarityImpl;
      this.theUser = theUser;
      this.minSim = minSim;
    }

    @Override
    public double estimate(User user) throws TasteException {
      if (user.equals(theUser)) {
        return Double.NaN;
      }
      double sim = userSimilarityImpl.userSimilarity(theUser, user);
      return (sim >= minSim) ? sim : Double.NaN; 
    }
  }
}
