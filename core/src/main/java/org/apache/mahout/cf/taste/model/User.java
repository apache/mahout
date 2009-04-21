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

/**
 * <p>Implementations represent a user, who has preferences for {@link Item}s.</p>
 */
public interface User extends Comparable<User> {

  /**
   * @return unique user ID
   */
  Object getID();

  /**
   * @param itemID ID of item to get the user's preference for
   * @return user's {@link Preference} for that {@link Item}, or <code>null</code> if the user expresses
   *         no such preference
   */
  Preference getPreferenceFor(Object itemID);

  /**
   * Sets a preference that this {@link User} has. Note that in general callers should expect this to
   * be a slow operation, compared to {@link #getPreferenceFor(Object)}.
   */
  void setPreference(Item item, double value);

  /**
   * Removes a preference. This method should also be considered potentially slow.
   */
  void removePreference(Object itemID);

  /**
   * <p>Returns a sequence of {@link Preference}s for this {@link User} which can be iterated over.
   * Note that the sequence <em>must</em> be "in order": ordered by {@link Item}.</p>
   *
   * @return a sequence of {@link Preference}s
   */
  Iterable<Preference> getPreferences();

  /**
   * <p>Returns an array view of {@link Preference}s for this {@link User}.
   * Note that the sequence <em>must</em> be "in order": ordered by {@link Item}.</p>
   *
   * @return an array of {@link Preference}s
   */
  Preference[] getPreferencesAsArray();

}
