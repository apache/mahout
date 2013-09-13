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

import java.util.List;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;

/**
 * <p>
 * Implementations of this interface can recommend items for a user. Implementations will likely take
 * advantage of several classes in other packages here to compute this.
 * </p>
 */
public interface Recommender extends Refreshable {
  
  /**
   * @param userID
   *          user for which recommendations are to be computed
   * @param howMany
   *          desired number of recommendations
   * @return {@link List} of recommended {@link RecommendedItem}s, ordered from most strongly recommend to
   *         least
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  List<RecommendedItem> recommend(long userID, int howMany) throws TasteException;
  
  /**
   * @param userID
   *          user for which recommendations are to be computed
   * @param howMany
   *          desired number of recommendations
   * @param rescorer
   *          rescoring function to apply before final list of recommendations is determined
   * @return {@link List} of recommended {@link RecommendedItem}s, ordered from most strongly recommend to
   *         least
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException;
  
  /**
   * @param userID
   *          user ID whose preference is to be estimated
   * @param itemID
   *          item ID to estimate preference for
   * @return an estimated preference if the user has not expressed a preference for the item, or else the
   *         user's actual preference for the item. If a preference cannot be estimated, returns
   *         {@link Double#NaN}
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  float estimatePreference(long userID, long itemID) throws TasteException;
  
  /**
   * @param userID
   *          user to set preference for
   * @param itemID
   *          item to set preference for
   * @param value
   *          preference value
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  void setPreference(long userID, long itemID, float value) throws TasteException;
  
  /**
   * @param userID
   *          user from which to remove preference
   * @param itemID
   *          item for which to remove preference
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel}
   */
  void removePreference(long userID, long itemID) throws TasteException;

  /**
   * @return underlying {@link DataModel} used by this {@link Recommender} implementation
   */
  DataModel getDataModel();

}
