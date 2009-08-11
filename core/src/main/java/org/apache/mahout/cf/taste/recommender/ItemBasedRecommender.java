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

import java.util.List;

/** <p>Interface implemented by "item-based" recommenders.</p> */
public interface ItemBasedRecommender extends Recommender {

  /**
   * @param itemID  ID of item for which to find most similar other items
   * @param howMany desired number of most similar items to find
   * @return items most similar to the given item, ordered from most similar to least
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<RecommendedItem> mostSimilarItems(long itemID, int howMany) throws TasteException;

  /**
   * @param itemID   ID of item for which to find most similar other items
   * @param howMany  desired number of most similar items to find
   * @param rescorer {@link Rescorer} which can adjust item-item similarity estimates used to determine most similar
   *                 items
   * @return itemss most similar to the given item, ordered from most similar to least
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<RecommendedItem> mostSimilarItems(long itemID,
                                         int howMany,
                                         Rescorer<LongPair> rescorer) throws TasteException;

  /**
   * @param itemIDs IDs of item for which to find most similar other items
   * @param howMany desired number of most similar items to find estimates used to determine most similar items
   * @return items most similar to the given items, ordered from most similar to least
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<RecommendedItem> mostSimilarItems(long[] itemIDs, int howMany) throws TasteException;

  /**
   * @param itemIDs  IDs of item for which to find most similar other items
   * @param howMany  desired number of most similar items to find
   * @param rescorer {@link Rescorer} which can adjust item-item similarity estimates used to determine most similar
   *                 items
   * @return items most similar to the given items, ordered from most similar to least
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<RecommendedItem> mostSimilarItems(long[] itemIDs,
                                         int howMany,
                                         Rescorer<LongPair> rescorer) throws TasteException;

  /**
   * <p>Lists the items that were most influential in recommending a given item to a given user. Exactly how
   * this is determined is left to the implementation, but, generally this will return items that the user prefers and
   * that are similar to the given item.</p>
   *
   * <p>This returns a {@link List} of {@link RecommendedItem} which is a little misleading since it's returning
   * recommend<strong>ing</strong> items, but, I thought it more natural to just reuse this class since it encapsulates
   * an item and value. The value here does not necessarily have a consistent interpretation or expected range;
   * it will be higher the more influential the item was in the recommendation.</p>
   *
   * @param userID  ID of user who was recommended the item
   * @param itemID  ID of item that was recommended
   * @param howMany maximum number of items to return
   * @return {@link List} of {@link RecommendedItem}, ordered from most influential in recommended the
   * given item to least
   * @throws TasteException if an error occurs while accessing the {@link org.apache.mahout.cf.taste.model.DataModel}
   */
  List<RecommendedItem> recommendedBecause(long userID, long itemID, int howMany) throws TasteException;

}
