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

package org.apache.mahout.cf.taste.impl.model;

import com.google.common.base.Preconditions;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * This is a special thread-safe version of {@link PlusAnonymousUserDataModel}
 * which allow multiple concurrent anonymous requests.
 * </p>
 *
 * <p>
 * To use it, you have to estimate the number of concurrent anonymous users of your application.
 * The pool of users with the given size will be created. For each anonymous recommendations request,
 * a user has to be taken from the pool and returned back immediately afterwards.
 * </p>
 *
 * <p>
 * If no more users are available in the pool, anonymous recommendations cannot be produced.
 * </p>
 *
 * </p>
 *
 * Setup:
 * <pre>
 * int concurrentUsers = 100;
 * DataModel realModel = ..
 * PlusAnonymousConcurrentUserDataModel plusModel =
 *   new PlusAnonymousConcurrentUserDataModel(realModel, concurrentUsers);
 * Recommender recommender = ...;
 * </pre>
 *
 * Real-time recommendation:
 * <pre>
 * PlusAnonymousConcurrentUserDataModel plusModel =
 *   (PlusAnonymousConcurrentUserDataModel) recommender.getDataModel();
 *
 * // Take the next available anonymous user from the pool
 * Long anonymousUserID = plusModel.takeAvailableUser();
 *
 * PreferenceArray tempPrefs = ..
 * tempPrefs.setUserID(0, anonymousUserID);
 * tempPrefs.setItemID(0, itemID);
 * plusModel.setTempPrefs(tempPrefs, anonymousUserID);
 *
 * // Produce recommendations
 * recommender.recommend(anonymousUserID, howMany);
 *
 * // It is very IMPORTANT to release user back to the pool
 * plusModel.releaseUser(anonymousUserID);
 * </pre>
 *
 * </p>
 */
public final class PlusAnonymousConcurrentUserDataModel extends PlusAnonymousUserDataModel {

  /** Preferences for all anonymous users */
  private final Map<Long,PreferenceArray> tempPrefs;
  /** Item IDs set for all anonymous users */
  private final Map<Long,FastIDSet> prefItemIDs;
  /** Pool of the users (FIFO) */
  private Queue<Long> usersPool;

  private static final Logger log = LoggerFactory.getLogger(PlusAnonymousUserDataModel.class);

  /**
   * @param delegate Real model where anonymous users will be added to
   * @param maxConcurrentUsers Maximum allowed number of concurrent anonymous users
   */
  public PlusAnonymousConcurrentUserDataModel(DataModel delegate, int maxConcurrentUsers) {
    super(delegate);

    tempPrefs = new ConcurrentHashMap<Long, PreferenceArray>();
    prefItemIDs = new ConcurrentHashMap<Long, FastIDSet>();

    initializeUsersPools(maxConcurrentUsers);
  }

  /**
   * Initialize the pool of concurrent anonymous users.
   *
   * @param usersPoolSize Maximum allowed number of concurrent anonymous user. Depends on the consumer system.
   */
  private void initializeUsersPools(int usersPoolSize) {
    usersPool = new ConcurrentLinkedQueue<Long>();
    for (int i = 0; i < usersPoolSize; i++) {
      usersPool.add(TEMP_USER_ID + i);
    }
  }

  /**
   * Take the next available concurrent anonymous users from the pool.
   *
   * @return User ID or null if no more users are available
   */
  public Long takeAvailableUser() {
    Long takenUserID = usersPool.poll();
    if (takenUserID != null) {
      // Initialize the preferences array to indicate that the user is taken.
      tempPrefs.put(takenUserID, new GenericUserPreferenceArray(0));
      return takenUserID;
    }
    return null;
  }

  /**
   * Release previously taken anonymous user and return it to the pool.
   *
   * @param userID ID of a previously taken anonymous user
   * @return true if the user was previously taken, false otherwise
   */
  public boolean releaseUser(Long userID) {
    if (tempPrefs.containsKey(userID)) {
      this.clearTempPrefs(userID);
      // Return previously taken user to the pool
      usersPool.offer(userID);
      return true;
    }
    return false;
  }

  /**
   * Checks whether a given user is a valid previously acquired anonymous user.
   */
  private boolean isAnonymousUser(long userID) {
    return tempPrefs.containsKey(userID);
  }

  /**
   * Sets temporary preferences for a given anonymous user.
   */
  public void setTempPrefs(PreferenceArray prefs, long anonymousUserID) {
    Preconditions.checkArgument(prefs != null && prefs.length() > 0, "prefs is null or empty");

    this.tempPrefs.put(anonymousUserID, prefs);
    FastIDSet userPrefItemIDs = new FastIDSet();

    for (int i = 0; i < prefs.length(); i++) {
      userPrefItemIDs.add(prefs.getItemID(i));
    }

    this.prefItemIDs.put(anonymousUserID, userPrefItemIDs);
  }

  /**
   * Clears temporary preferences for a given anonymous user.
   */
  public void clearTempPrefs(long anonymousUserID) {
    this.tempPrefs.remove(anonymousUserID);
    this.prefItemIDs.remove(anonymousUserID);
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    // Anonymous users have short lifetime and should not be included into the neighbohoods of the real users.
    // Thus exclude them from the universe.
    return getDelegate().getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    if (isAnonymousUser(userID)) {
      return tempPrefs.get(userID);
    }
    return getDelegate().getPreferencesFromUser(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    if (isAnonymousUser(userID)) {
      return prefItemIDs.get(userID);
    }
    return getDelegate().getItemIDsFromUser(userID);
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    if (tempPrefs.isEmpty()) {
      return getDelegate().getPreferencesForItem(itemID);
    }

    PreferenceArray delegatePrefs = null;

    try {
      delegatePrefs = getDelegate().getPreferencesForItem(itemID);
    } catch (NoSuchItemException nsie) {
      // OK. Probably an item that only the anonymous user has
      if (log.isDebugEnabled()) {
        log.debug("Item {} unknown", itemID);
      }
    }

    List<Preference> anonymousPreferences = Lists.newArrayList();

    for (Map.Entry<Long, PreferenceArray> prefsMap : tempPrefs.entrySet()) {
      PreferenceArray singleUserTempPrefs = prefsMap.getValue();
      for (int i = 0; i < singleUserTempPrefs.length(); i++) {
        if (singleUserTempPrefs.getItemID(i) == itemID) {
          anonymousPreferences.add(singleUserTempPrefs.get(i));
        }
      }
    }

    int delegateLength = delegatePrefs == null ? 0 : delegatePrefs.length();
    int anonymousPrefsLength = anonymousPreferences.size();
    int prefsCounter = 0;

    // Merge the delegate and anonymous preferences into a single array
    PreferenceArray newPreferenceArray = new GenericItemPreferenceArray(delegateLength + anonymousPrefsLength);

    for (int i = 0; i < delegateLength; i++) {
      newPreferenceArray.set(prefsCounter++, delegatePrefs.get(i));
    }

    for (Preference anonymousPreference : anonymousPreferences) {
      newPreferenceArray.set(prefsCounter++, anonymousPreference);
    }

    if (newPreferenceArray.length() == 0) {
      // No, didn't find it among the anonymous user prefs
      throw new NoSuchItemException(itemID);
    }

    return newPreferenceArray;
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    if (isAnonymousUser(userID)) {
      PreferenceArray singleUserTempPrefs = tempPrefs.get(userID);
      for (int i = 0; i < singleUserTempPrefs.length(); i++) {
        if (singleUserTempPrefs.getItemID(i) == itemID) {
          return singleUserTempPrefs.getValue(i);
        }
      }
      return null;
    }
    return getDelegate().getPreferenceValue(userID, itemID);
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    if (isAnonymousUser(userID)) {
      // Timestamps are not saved for anonymous preferences
      return null;
    }
    return getDelegate().getPreferenceTime(userID, itemID);
  }

  @Override
  public int getNumUsers() throws TasteException {
    // Anonymous users have short lifetime and should not be included into the neighbohoods of the real users.
    // Thus exclude them from the universe.
    return getDelegate().getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    if (tempPrefs.isEmpty()) {
      return getDelegate().getNumUsersWithPreferenceFor(itemID);
    }

    int countAnonymousUsersWithPreferenceFor = 0;

    for (Map.Entry<Long, PreferenceArray> singleUserTempPrefs : tempPrefs.entrySet()) {
      for (int i = 0; i < singleUserTempPrefs.getValue().length(); i++) {
        if (singleUserTempPrefs.getValue().getItemID(i) == itemID) {
          countAnonymousUsersWithPreferenceFor++;
          break;
        }
      }
    }
    return getDelegate().getNumUsersWithPreferenceFor(itemID) + countAnonymousUsersWithPreferenceFor;
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    if (tempPrefs.isEmpty()) {
      return getDelegate().getNumUsersWithPreferenceFor(itemID1, itemID2);
    }

    int countAnonymousUsersWithPreferenceFor = 0;

    for (Map.Entry<Long, PreferenceArray> singleUserTempPrefs : tempPrefs.entrySet()) {
      boolean found1 = false;
      boolean found2 = false;
      for (int i = 0; i < singleUserTempPrefs.getValue().length() && !(found1 && found2); i++) {
        long itemID = singleUserTempPrefs.getValue().getItemID(i);
        if (itemID == itemID1) {
          found1 = true;
        }
        if (itemID == itemID2) {
          found2 = true;
        }
      }

      if (found1 && found2) {
        countAnonymousUsersWithPreferenceFor++;
      }
    }

    return getDelegate().getNumUsersWithPreferenceFor(itemID1, itemID2) + countAnonymousUsersWithPreferenceFor;
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    if (isAnonymousUser(userID)) {
      throw new UnsupportedOperationException();
    }
    getDelegate().setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    if (isAnonymousUser(userID)) {
      throw new UnsupportedOperationException();
    }
    getDelegate().removePreference(userID, itemID);
  }
}
