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

package org.apache.mahout.cf.taste.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;

import java.util.Collection;

/** <p>Interface implemented by "clustering" recommenders.</p> */
public interface ClusteringRecommender extends Recommender {

  /**
   * <p>Returns the cluster of users to which the given user, denoted by user ID, belongs.</p>
   *
   * @param userID user ID for which to find a cluster
   * @return {@link Collection} of IDs of users in the requested user's cluster
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  FastIDSet getCluster(long userID) throws TasteException;

  /**
   * <p>Returns all clusters of users.</p>
   *
   * @return {@link Collection} of {@link Collection}s of user IDs
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  FastIDSet[] getClusters() throws TasteException;

}
