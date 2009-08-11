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
import org.apache.mahout.cf.taste.impl.common.LongPair;

/** <p>Interface implemented by "user-based" recommenders.</p> */
public interface UserBasedRecommender extends Recommender {

  /**
   * @param userID  ID of user for which to find most similar other users
   * @param howMany desired number of most similar users to find
   * @return users most similar to the given user
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  long[] mostSimilarUserIDs(long userID, int howMany) throws TasteException;

  /**
   * @param userID   ID of user for which to find most similar other users
   * @param howMany  desired number of most similar users to find
   * @param rescorer {@link Rescorer} which can adjust user-user similarity estimates used to determine most similar
   *                 users
   * @return IDs of users most similar to the given user
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  long[] mostSimilarUserIDs(long userID, int howMany, Rescorer<LongPair> rescorer) throws TasteException;

}
