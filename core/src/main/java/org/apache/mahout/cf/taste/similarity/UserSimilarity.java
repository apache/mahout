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

package org.apache.mahout.cf.taste.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>Implementations of this interface define a notion of similarity between two users. Implementations should
 * return values in the range -1.0 to 1.0, with 1.0 representing perfect similarity.</p>
 *
 * @see ItemSimilarity
 */
public interface UserSimilarity extends Refreshable {

  /**
   * <p>Returns the degree of similarity, of two users, based on the their preferences.</p>
   *
   * @param userID1 first user ID
   * @param userID2 second user ID
   * @return similarity between the two users, in [-1,1]
   * @throws TasteException if an error occurs while accessing the data
   */
  double userSimilarity(long userID1, long userID2) throws TasteException;

  /**
   * <p>Attaches a {@link PreferenceInferrer} to the {@link UserSimilarity} implementation.</p>
   *
   * @param inferrer {@link PreferenceInferrer}
   */
  void setPreferenceInferrer(PreferenceInferrer inferrer);

}
