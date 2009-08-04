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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * <p>A simple {@link DataModel} which uses a given {@link List} of users as its data source. This
 * implementation is mostly useful for small experiments and is not recommended for contexts where performance is
 * important.</p>
 */
public final class GenericDataModel implements DataModel, Serializable {

  private static final Logger log = LoggerFactory.getLogger(GenericDataModel.class);

  private final List<Comparable<?>> userIDs;
  private final Map<Comparable<?>, PreferenceArray> preferenceFromUsers;
  private final List<Comparable<?>> itemIDs;
  private final Map<Comparable<?>, PreferenceArray> preferenceForItems;

  /**
   * <p>Creates a new {@link GenericDataModel} from the given users (and their preferences). This {@link
   * DataModel} retains all this information in memory and is effectively immutable.</p>
   *
   * @param userData users to include in this {@link GenericDataModel}
   *  (see also {@link #toPrefArrayValues(Map, boolean)})
   */
  @SuppressWarnings("unchecked")
  public GenericDataModel(Map<Comparable<?>, PreferenceArray> userData) {
    if (userData == null) {
      throw new IllegalArgumentException("userData is null");
    }

    this.preferenceFromUsers = userData;
    FastMap<Comparable<?>, Collection<Preference>> prefsForItems = new FastMap<Comparable<?>, Collection<Preference>>();
    FastSet<Comparable<?>> itemIDSet = new FastSet<Comparable<?>>();
    int currentCount = 0;
    for (PreferenceArray prefs : preferenceFromUsers.values()) {
      prefs.sortByItem();
      int size = prefs.length();
      for (Preference preference : prefs) {
        Comparable<?> itemID = preference.getItemID();
        itemIDSet.add(itemID);
        List<Preference> prefsForItem = (List<Preference>) prefsForItems.get(itemID);
        if (prefsForItem == null) {
          prefsForItem = new ArrayList<Preference>(2);
          prefsForItems.put(itemID, prefsForItem);
        }
        prefsForItem.add(preference);
      }
      currentCount++;
      if (currentCount % 10000 == 0) {
        log.info("Processed {} users", currentCount);
      }
    }

    this.itemIDs = new ArrayList<Comparable<?>>(itemIDSet);
    itemIDSet = null;
    Collections.sort((List<? extends Comparable>) this.itemIDs);

    this.preferenceForItems = toPrefArrayValues(prefsForItems, false);

    for (PreferenceArray prefs : preferenceForItems.values()) {
      prefs.sortByUser();
    }
    
    this.userIDs = new ArrayList(userData.keySet());
    Collections.sort((List<? extends Comparable>) userIDs);
  }

  /**
   * <p>Creates a new {@link GenericDataModel} containing an immutable copy of the data from another given {@link
   * DataModel}.</p>
   *
   * @param dataModel {@link DataModel} to copy
   * @throws TasteException if an error occurs while retrieving the other {@link DataModel}'s users
   */
  public GenericDataModel(DataModel dataModel) throws TasteException {
    this(toDataMap(dataModel));
  }

  /**
   * Swaps, in-place, {@link List}s for arrays in {@link Map} values
   * .
   * @return input value
   */
  public static Map<Comparable<?>, PreferenceArray> toPrefArrayValues(Map<Comparable<?>, Collection<Preference>> data,
                                                                      boolean byUser) {
    for (Map.Entry<Comparable<?>, Object> entry :
         ((Map<Comparable<?>, Object>) (Map<Comparable<?>, ?>) data).entrySet()) {
      List<Preference> prefList = (List<Preference>) entry.getValue();
      entry.setValue(byUser ? new GenericUserPreferenceArray(prefList) : new GenericItemPreferenceArray(prefList));
    }
    return (Map<Comparable<?>, PreferenceArray>) (Map<Comparable<?>, ?>) data;
  }

  private static Map<Comparable<?>, PreferenceArray> toDataMap(DataModel dataModel) throws TasteException {
    Map<Comparable<?>, PreferenceArray> data = new FastMap<Comparable<?>, PreferenceArray>(dataModel.getNumUsers());
    for (Comparable<?> userID : dataModel.getUserIDs()) {
      data.put(userID, dataModel.getPreferencesFromUser(userID));
    }
    return data;
  }

  @Override
  public Iterable<Comparable<?>> getUserIDs() {
    return userIDs;
  }

  /** @throws NoSuchUserException if there is no such user */
  @Override
  public PreferenceArray getPreferencesFromUser(Comparable<?> userID) throws NoSuchUserException {
    PreferenceArray prefs = preferenceFromUsers.get(userID);
    if (prefs == null) {
      throw new NoSuchUserException();
    }
    return prefs;
  }

  @Override
  public FastSet<Comparable<?>> getItemIDsFromUser(Comparable<?> userID) throws TasteException {
    PreferenceArray prefs = getPreferencesFromUser(userID);
    int size = prefs.length();
    FastSet<Comparable<?>> result = new FastSet<Comparable<?>>(size);
    for (int i = 0; i < size; i++) {
      result.add(prefs.getItemID(i));
    }
    return result;
  }

  @Override
  public Iterable<Comparable<?>> getItemIDs() {
    return itemIDs;
  }

  @Override
  public PreferenceArray getPreferencesForItem(Comparable<?> itemID) throws NoSuchItemException {
    PreferenceArray prefs = preferenceForItems.get(itemID);
    if (prefs == null) {
      throw new NoSuchItemException();
    }
    return prefs;
  }

  @Override
  public Float getPreferenceValue(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
    PreferenceArray prefs = getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      if (prefs.getItemID(i).equals(itemID)) {
        return prefs.getValue(i);
      }
    }
    return null;
  }

  @Override
  public int getNumItems() {
    return itemIDs.size();
  }

  @Override
  public int getNumUsers() {
    return userIDs.size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(Comparable<?>... itemIDs) {
    if (itemIDs == null) {
      throw new IllegalArgumentException("itemIDs is null");
    }
    int length = itemIDs.length;
    if (length == 0 || length > 2) {
      throw new IllegalArgumentException("Illegal number of item IDs: " + length);
    }
    if (length == 1) {
      PreferenceArray prefs = preferenceForItems.get(itemIDs[0]);
      return prefs == null ? 0 : prefs.length();
    } else {
      PreferenceArray prefs1 = preferenceForItems.get(itemIDs[0]);
      PreferenceArray prefs2 = preferenceForItems.get(itemIDs[1]);
      if (prefs1 == null || prefs2 == null) {
        return 0;
      }
      Set<Comparable<?>> users1 = new FastSet<Comparable<?>>(prefs1.length());
      int size1 = prefs1.length();
      for (int i = 0; i < size1; i++) {
        users1.add(prefs1.getUserID(i));
      }
      Set<Comparable<?>> users2 = new FastSet<Comparable<?>>(prefs2.length());
      int size2 = prefs2.length();
      for (int i = 0; i < size2; i++) {
        users2.add(prefs2.getUserID(i));
      }
      users1.retainAll(users2);
      return users1.size();
    }
  }

  @Override
  public void removePreference(Comparable<?> userID, Comparable<?> itemID) throws NoSuchUserException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, float value) throws NoSuchUserException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Does nothing
  }

  @Override
  public String toString() {
    return "GenericDataModel[users:" + (userIDs.size() > 3 ? userIDs.subList(0, 3) + "..." : userIDs) + ']';
  }

}
