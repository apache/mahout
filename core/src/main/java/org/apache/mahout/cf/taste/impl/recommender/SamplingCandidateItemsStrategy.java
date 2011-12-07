/*
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
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveArrayIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.FixedSizeSamplingIterator;

import java.util.Iterator;

/**
 * <p>Returns all items that have not been rated by the user <em>(3)</em> and that were preferred by another user
 * <em>(2)</em> that has preferred at least one item <em>(1)</em> that the current user has preferred too.</p>
 *
 * <p>This strategy uses sampling to limit the number of items that are considered, by sampling three different
 * things, noted above:</p>
 *
 * <ol>
 *   <li>The items</li>
 * </ol>
 * 
 * <p><pre>
 * max(defaultMaxPrefsPerItemConsidered, userItemCountFactor * log(max(N_users, N_items)))
 * </pre></p>
 * 
 * <p>This limit is applied in two ways. First, for each item that the current user prefers, it limits
 * the number of other users who preferred that item that are then considered. Then for each of those users,
 * it limits the number of their preferred items that are added to the list of candidates.</p>
 */
public class SamplingCandidateItemsStrategy extends AbstractCandidateItemsStrategy {

  private static final int DEFAULT_FACTOR = 5;
  
  private final int maxItems;
  private final int maxUsersPerItem;
  private final int maxItemsPerUser;

  public SamplingCandidateItemsStrategy(int numUsers, int numItems) {
    this(DEFAULT_FACTOR, DEFAULT_FACTOR, DEFAULT_FACTOR, numUsers, numItems);
  }

  public SamplingCandidateItemsStrategy(int itemsFactor,
                                        int usersPerItemFactor,
                                        int candidatesPerUserFactor,
                                        int numUsers,
                                        int numItems) {
    Preconditions.checkArgument(itemsFactor > 0);
    Preconditions.checkArgument(usersPerItemFactor > 0);
    Preconditions.checkArgument(candidatesPerUserFactor > 0);
    Preconditions.checkArgument(numUsers > 0);
    Preconditions.checkArgument(numItems > 0);
    maxItems = (int) (itemsFactor * (1.0 + Math.log(numItems)));
    maxUsersPerItem = (int) (itemsFactor * (1.0 + Math.log(numUsers)));
    maxItemsPerUser = (int) (itemsFactor *(1.0 + Math.log(numItems)));
  }

  @Override
  protected FastIDSet doGetCandidateItems(long[] preferredItemIDs, DataModel dataModel) throws TasteException {
    FastIDSet possibleItemsIDs = new FastIDSet();
    LongPrimitiveIterator preferredItemIDsIterator = new LongPrimitiveArrayIterator(preferredItemIDs);
    if (preferredItemIDs.length > maxItems) {
      preferredItemIDsIterator =
          new SamplingLongPrimitiveIterator(preferredItemIDsIterator, (double) maxItems / preferredItemIDs.length);
    }
    while (preferredItemIDsIterator.hasNext()) {
      long itemID = preferredItemIDsIterator.nextLong();
      PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
      int prefsLength = prefs.length();
      if (prefsLength > maxUsersPerItem) {
        Iterator<Preference> sampledPrefs =
            new FixedSizeSamplingIterator<Preference>(maxUsersPerItem, prefs.iterator());
        while (sampledPrefs.hasNext()) {
          addSomeOf(possibleItemsIDs, dataModel.getItemIDsFromUser(sampledPrefs.next().getUserID()));
        }
      } else {
        for (int i = 0; i < prefsLength; i++) {
          addSomeOf(possibleItemsIDs, dataModel.getItemIDsFromUser(prefs.getUserID(i)));
        }
      }
    }
    possibleItemsIDs.removeAll(preferredItemIDs);
    return possibleItemsIDs;
  }
  
  private void addSomeOf(FastIDSet possibleItemIDs, FastIDSet itemIDs) {
    if (itemIDs.size() > maxItemsPerUser) {
      LongPrimitiveIterator it =
          new SamplingLongPrimitiveIterator(itemIDs.iterator(), (double) maxItemsPerUser / itemIDs.size());
      while (it.hasNext()) {
        possibleItemIDs.add(it.nextLong());
      }
    } else {
      possibleItemIDs.addAll(itemIDs);
    }
  }
  
}
