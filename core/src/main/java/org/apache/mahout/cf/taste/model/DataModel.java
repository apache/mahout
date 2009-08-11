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

package org.apache.mahout.cf.taste.model;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

import java.util.List;

/**
 * <p>Implementations represent a repository of information about users and their associated {@link Preference}s
 * for items.</p>
 */
public interface DataModel extends Refreshable {

  /**
   * @return all user IDs in the model, in order
   * @throws TasteException if an error occurs while accessing the data
   */
  LongPrimitiveIterator getUserIDs() throws TasteException;

  /**
   * @param userID ID of user to get prefs for
   * @return user's preferences
   * @throws NoSuchUserException if the user does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  PreferenceArray getPreferencesFromUser(long userID) throws TasteException;

  /**
   * @param userID ID of user to get prefs for
   * @return IDs of items user expresses a preference for
   * @throws NoSuchUserException if the user does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  FastIDSet getItemIDsFromUser(long userID) throws TasteException;

  /**
   * @return a {@link List} of all item IDs in the model, in order
   * @throws TasteException if an error occurs while accessing the data
   */
  LongPrimitiveIterator getItemIDs() throws TasteException;

  /**
   * @param itemID item ID
   * @return all existing {@link Preference}s expressed for that item, ordered by user ID, as an array
   * @throws NoSuchItemException if the item does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  PreferenceArray getPreferencesForItem(long itemID) throws TasteException;

  /**
   * Retrieves the preference value for a single user and item.
   *
   * @param userID user ID to get pref value from
   * @param itemID item ID to get pref value for
   * @return preference value from the given user for the given item or null if none exists
   * @throws NoSuchUserException if the user does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  Float getPreferenceValue(long userID, long itemID) throws TasteException;

  /**
   * @return total number of items known to the model. This is generally the union of all items
   *         preferred by at least one user but could include more.
   * @throws TasteException if an error occurs while accessing the data
   */
  int getNumItems() throws TasteException;

  /**
   * @return total number of users known to the model.
   * @throws TasteException if an error occurs while accessing the data
   */
  int getNumUsers() throws TasteException;

  /**
   * @param itemIDs item IDs to check for
   * @return the number of users who have expressed a preference for all of the items
   * @throws TasteException           if an error occurs while accessing the data
   * @throws IllegalArgumentException if itemIDs is null, empty, or larger than 2 elements since currently only queries
   *                                  of up to 2 items are needed and supported
   * @throws NoSuchItemException if an item does not exist
   */
  int getNumUsersWithPreferenceFor(long... itemIDs) throws TasteException;

  /**
   * <p>Sets a particular preference (item plus rating) for a user.</p>
   *
   * @param userID user to set preference for
   * @param itemID item to set preference for
   * @param value  preference value
   * @throws NoSuchItemException if the item does not exist
   * @throws NoSuchUserException if the user does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  void setPreference(long userID, long itemID, float value) throws TasteException;

  /**
   * <p>Removes a particular preference for a user.</p>
   *
   * @param userID user from which to remove preference
   * @param itemID item to remove preference for
   * @throws NoSuchItemException if the item does not exist
   * @throws NoSuchUserException if the user does not exist
   * @throws TasteException if an error occurs while accessing the data
   */
  void removePreference(long userID, long itemID) throws TasteException;

}
