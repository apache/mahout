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
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.model.User;

import java.util.List;

/**
 * <p>Interface implemented by "user-based" recommenders.</p>
 */
public interface UserBasedRecommender extends Recommender {

  /**
   * @param userID ID of {@link User} for which to find most similar other {@link User}s
   * @param howMany desired number of most similar {@link User}s to find
   * @return {@link User}s most similar to the given user
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<User> mostSimilarUsers(Object userID, int howMany) throws TasteException;

  /**
   * @param userID ID of {@link User} for which to find most similar other {@link User}s
   * @param howMany desired number of most similar {@link User}s to find
   * @param rescorer {@link Rescorer} which can adjust user-user correlation
   * estimates used to determine most similar users
   * @return {@link User}s most similar to the given user
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<User> mostSimilarUsers(Object userID, int howMany, Rescorer<Pair<User, User>> rescorer) throws TasteException;

}
