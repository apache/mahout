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

package org.apache.mahout.cf.taste.recommender.slopeone;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * <p>
 * Implementations store item-item preference diffs for a
 * {@link org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender}. It actually does a bit
 * more for this implementation, like listing all items that may be considered for recommendation, in order to
 * maximize what implementations can do to optimize the slope-one algorithm.
 * </p>
 * 
 * @see org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender
 */
public interface DiffStorage extends Refreshable {
  
  /**
   * @return {@link RunningAverage} encapsulating the average difference in preferences between items
   *         corresponding to {@code itemID1} and {@code itemID2}, in that direction; that is, it's
   *         the average of item 2's preferences minus item 1's preferences
   */
  RunningAverage getDiff(long itemID1, long itemID2) throws TasteException;
  
  /**
   * @param userID
   *          user ID to get diffs for
   * @param itemID
   *          itemID to assess
   * @param prefs
   *          user's preferendces
   * @return {@link RunningAverage}s for that user's item-item diffs
   */
  RunningAverage[] getDiffs(long userID, long itemID, PreferenceArray prefs) throws TasteException;
  
  /** @return {@link RunningAverage} encapsulating the average preference for the given item */
  RunningAverage getAverageItemPref(long itemID) throws TasteException;

  /**
   * <p>Updates internal data structures to reflect a new preference value for an item.</p>
   *
   * @param userID user whose pref is being added
   * @param itemID item to add preference value for
   * @param prefValue new preference value
   */
  void addItemPref(long userID, long itemID, float prefValue) throws TasteException;

  /**
   * <p>Updates internal data structures to reflect an update in a preference value for an item.</p>
   * 
   * @param itemID item to update preference value for
   * @param prefDelta amount by which preference value changed
   */
  void updateItemPref(long itemID, float prefDelta) throws TasteException;

  /**
   * <p>Updates internal data structures to reflect an update in a preference value for an item.</p>
   *
   * @param userID user whose pref is being removed
   * @param itemID item to update preference value for
   * @param prefValue old preference value
   */
  void removeItemPref(long userID, long itemID, float prefValue) throws TasteException;
  
  /**
   * @return item IDs that may possibly be recommended to the given user, which may not be all items since the
   *         item-item diff matrix may be sparse
   */
  FastIDSet getRecommendableItemIDs(long userID) throws TasteException;
  
}
