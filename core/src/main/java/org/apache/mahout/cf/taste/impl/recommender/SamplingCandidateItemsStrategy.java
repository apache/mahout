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

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.FixedSizeSamplingIterator;

import java.util.Iterator;

/**
 * <p>returns all items that have not been rated by the user and that were preferred by another user
 * that has preferred at least one item that the current user has preferred too</p>
 *
 * <p>this strategy uses sampling in a way that only a certain amount of preferences per item is considered
 * <pre>
 * max(defaultMaxPrefsPerItemConsidered, userItemCountFactor * log(max(N_users, N_items)))
 * </pre></p>
 */
public class SamplingCandidateItemsStrategy extends AbstractCandidateItemsStrategy {

  private final int defaultMaxPrefsPerItemConsidered;
  private final int userItemCountMultiplier;

  /**
   * uses defaultMaxPrefsPerItemConsidered = 100 and userItemCountMultiplier = 20 as default values
   * 
   * @see SamplingCandidateItemsStrategy#SamplingCandidateItemsStrategy(int, int)
   */
  public SamplingCandidateItemsStrategy() {
    this(100, 20);
  }

  /**
   * <p>the maximum number of prefs considered per item will be computed like this:
   * <pre>
   *   max(defaultMaxPrefsPerItemConsidered, userItemCountFactor * log(max(N_users, N_items)))
   * </pre>
   * </p>
   */
  public SamplingCandidateItemsStrategy(int defaultMaxPrefsPerItemConsidered, int userItemCountMultiplier) {
    Preconditions.checkArgument(defaultMaxPrefsPerItemConsidered > 0, "defaultMaxPrefsPerItemConsidered must be " +
        "greater zero");
    Preconditions.checkArgument(userItemCountMultiplier > 0, "userItemCountMultiplier must be greater zero");
    this.defaultMaxPrefsPerItemConsidered = defaultMaxPrefsPerItemConsidered;
    this.userItemCountMultiplier = userItemCountMultiplier;
  }

  @Override
  protected FastIDSet doGetCandidateItems(long[] preferredItemIDs, DataModel dataModel) throws TasteException {
    int maxPrefsPerItemConsidered = (int) Math.max(defaultMaxPrefsPerItemConsidered,
        userItemCountMultiplier * Math.log(Math.max(dataModel.getNumUsers(), dataModel.getNumItems())));
    FastIDSet possibleItemsIDs = new FastIDSet();
    for (long itemID : preferredItemIDs) {
      PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
      int prefsConsidered = Math.min(prefs.length(), maxPrefsPerItemConsidered);
      Iterator<Preference> sampledPrefs = new FixedSizeSamplingIterator<Preference>(prefsConsidered, prefs.iterator());
      while (sampledPrefs.hasNext()) {
        possibleItemsIDs.addAll(dataModel.getItemIDsFromUser(sampledPrefs.next().getUserID()));
      }
    }
    possibleItemsIDs.removeAll(preferredItemIDs);
    return possibleItemsIDs;
  }
}
