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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;

/**
 * <p>Returns all items that have not been rated by the user <em>(3)</em> and that were preferred by another user
 * <em>(2)</em> that has preferred at least one item <em>(1)</em> that the current user has preferred too.</p>
 *
 * <p>This strategy uses sampling to limit the number of items that are considered, by sampling three different
 * things, noted above:</p>
 *
 * <ol>
 *   <li>The items that the user has preferred</li>
 *   <li>The users who also prefer each of those items</li>
 *   <li>The items those users also prefer</li>
 * </ol>
 * 
 * <p>There is a maximum associated with each of these three things; if the number of items or users exceeds
 * that max, it is sampled so that the expected number of items or users actually used in that part of the
 * computation is equal to the max.</p>
 * 
 * <p>Three arguments control these three maxima. Each is a "factor" f, which establishes the max at
 * f * log2(n), where n is the number of users or items in the data. For example if factor #2 is 5,
 * which controls the number of users sampled per item, then 5 * log2(# users) is the maximum for this
 * part of the computation.</p>
 * 
 * <p>Each can be set to not do any limiting with value {@link #NO_LIMIT_FACTOR}.</p>
 */
public class SamplingCandidateItemsStrategy extends AbstractCandidateItemsStrategy {

  private static final Logger log = LoggerFactory.getLogger(SamplingCandidateItemsStrategy.class);

  /**
   * Default factor used if not otherwise specified, for all limits. (30).
   */
  public static final int DEFAULT_FACTOR = 30;
  /**
   * Specify this value as a factor to mean no limit.
   */
  public static final int NO_LIMIT_FACTOR = Integer.MAX_VALUE;
  private static final int MAX_LIMIT = Integer.MAX_VALUE;
  private static final double LOG2 = Math.log(2.0);

  private final int maxItems;
  private final int maxUsersPerItem;
  private final int maxItemsPerUser;

  /**
   * Defaults to using no limit ({@link #NO_LIMIT_FACTOR}) for all factors, except 
   * {@code candidatesPerUserFactor} which defaults to {@link #DEFAULT_FACTOR}.
   *
   * @see #SamplingCandidateItemsStrategy(int, int, int, int, int)
   */
  public SamplingCandidateItemsStrategy(int numUsers, int numItems) {
    this(DEFAULT_FACTOR, DEFAULT_FACTOR, DEFAULT_FACTOR, numUsers, numItems);
  }

  /**
   * @param itemsFactor factor controlling max items considered for a user
   * @param usersPerItemFactor factor controlling max users considered for each of those items
   * @param candidatesPerUserFactor factor controlling max candidate items considered from each of those users
   * @param numUsers number of users currently in the data
   * @param numItems number of items in the data
   */
  public SamplingCandidateItemsStrategy(int itemsFactor,
                                        int usersPerItemFactor,
                                        int candidatesPerUserFactor,
                                        int numUsers,
                                        int numItems) {
    Preconditions.checkArgument(itemsFactor > 0, "itemsFactor must be greater then 0!");
    Preconditions.checkArgument(usersPerItemFactor > 0, "usersPerItemFactor must be greater then 0!");
    Preconditions.checkArgument(candidatesPerUserFactor > 0, "candidatesPerUserFactor must be greater then 0!");
    Preconditions.checkArgument(numUsers > 0, "numUsers must be greater then 0!");
    Preconditions.checkArgument(numItems > 0, "numItems must be greater then 0!");
    maxItems = computeMaxFrom(itemsFactor, numItems);
    maxUsersPerItem = computeMaxFrom(usersPerItemFactor, numUsers);
    maxItemsPerUser = computeMaxFrom(candidatesPerUserFactor, numItems);
    log.debug("maxItems {}, maxUsersPerItem {}, maxItemsPerUser {}", maxItems, maxUsersPerItem, maxItemsPerUser);
  }

  private static int computeMaxFrom(int factor, int numThings) {
    if (factor == NO_LIMIT_FACTOR) {
      return MAX_LIMIT;
    }
    long max = (long) (factor * (1.0 + Math.log(numThings) / LOG2));
    return max > MAX_LIMIT ? MAX_LIMIT : (int) max;
  }

  @Override
  protected FastIDSet doGetCandidateItems(long[] preferredItemIDs, DataModel dataModel) throws TasteException {
    LongPrimitiveIterator preferredItemIDsIterator = new LongPrimitiveArrayIterator(preferredItemIDs);
    if (preferredItemIDs.length > maxItems) {
      double samplingRate = (double) maxItems / preferredItemIDs.length;
//      log.info("preferredItemIDs.length {}, samplingRate {}", preferredItemIDs.length, samplingRate);
      preferredItemIDsIterator = 
          new SamplingLongPrimitiveIterator(preferredItemIDsIterator, samplingRate);
    }
    FastIDSet possibleItemsIDs = new FastIDSet();
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
