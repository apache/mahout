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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

import java.util.List;

/**
 * <p>Implementations represent a repository of information about {@link User}s and their associated {@link Preference}s
 * for items.</p>
 */
public interface DataModel extends Refreshable {

  /**
   * @return a {@link List} of all {@link User}s in the model, ordered by {@link User}
   * @throws TasteException if an error occurs while accessing the data
   */
  Iterable<? extends User> getUsers() throws TasteException;

  /**
   * @param id user ID
   * @return {@link User} who has that ID
   * @throws TasteException if an error occurs while accessing the data
   * @throws org.apache.mahout.cf.taste.common.NoSuchUserException
   *                        if there is no such {@link User}
   */
  User getUser(Comparable<?> id) throws TasteException;

  /**
   * @return a {@link List} of all item IDs in the model, in order
   * @throws TasteException if an error occurs while accessing the data
   */
  Iterable<Comparable<?>> getItemIDs() throws TasteException;

  /**
   * @param itemID item ID
   * @return all existing {@link Preference}s expressed for that item, ordered by {@link User}
   * @throws TasteException if an error occurs while accessing the data
   */
  Iterable<? extends Preference> getPreferencesForItem(Comparable<?> itemID) throws TasteException;

  /**
   * @param itemID item ID
   * @return all existing {@link Preference}s expressed for that item, ordered by {@link User}, as an array
   * @throws TasteException if an error occurs while accessing the data
   */
  Preference[] getPreferencesForItemAsArray(Comparable<?> itemID) throws TasteException;

  /**
   * @return total number of items known to the model. This is generally the union of all items
   *         preferred by at least one {@link User} but could include more.
   * @throws TasteException if an error occurs while accessing the data
   */
  int getNumItems() throws TasteException;

  /**
   * @return total number of {@link User}s known to the model.
   * @throws TasteException if an error occurs while accessing the data
   */
  int getNumUsers() throws TasteException;

  /**
   * @param itemIDs item IDs to check for
   * @return the number of users who have expressed a preference for all of the items
   * @throws TasteException           if an error occurs while accessing the data
   * @throws IllegalArgumentException if itemIDs is null, empty, or larger than 2 elements since currently only queries
   *                                  of up to 2 items are needed and supported
   */
  int getNumUsersWithPreferenceFor(Comparable<?>... itemIDs) throws TasteException;

  /**
   * <p>Sets a particular preference (item plus rating) for a user.</p>
   *
   * @param userID user to set preference for
   * @param itemID item to set preference for
   * @param value  preference value
   * @throws TasteException if an error occurs while accessing the data
   */
  void setPreference(Comparable<?> userID, Comparable<?> itemID, double value) throws TasteException;

  /**
   * <p>Removes a particular preference for a user.</p>
   *
   * @param userID user from which to remove preference
   * @param itemID item to remove preference for
   * @throws TasteException if an error occurs while accessing the data
   */
  void removePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException;

}
