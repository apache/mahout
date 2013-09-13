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

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveArrayIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * A simple {@link DataModel} which uses a given {@link List} of users as its data source. This implementation
 * is mostly useful for small experiments and is not recommended for contexts where performance is important.
 * </p>
 */
public final class GenericDataModel extends AbstractDataModel {
  
  private static final Logger log = LoggerFactory.getLogger(GenericDataModel.class);
  
  private final long[] userIDs;
  private final FastByIDMap<PreferenceArray> preferenceFromUsers;
  private final long[] itemIDs;
  private final FastByIDMap<PreferenceArray> preferenceForItems;
  private final FastByIDMap<FastByIDMap<Long>> timestamps;
  
  /**
   * <p>
   * Creates a new {@link GenericDataModel} from the given users (and their preferences). This
   * {@link DataModel} retains all this information in memory and is effectively immutable.
   * </p>
   * 
   * @param userData users to include; (see also {@link #toDataMap(FastByIDMap, boolean)})
   */
  public GenericDataModel(FastByIDMap<PreferenceArray> userData) {
    this(userData, null);
  }

  /**
   * <p>
   * Creates a new {@link GenericDataModel} from the given users (and their preferences). This
   * {@link DataModel} retains all this information in memory and is effectively immutable.
   * </p>
   *
   * @param userData users to include; (see also {@link #toDataMap(FastByIDMap, boolean)})
   * @param timestamps optionally, provided timestamps of preferences as milliseconds since the epoch.
   *  User IDs are mapped to maps of item IDs to Long timestamps.
   */
  public GenericDataModel(FastByIDMap<PreferenceArray> userData, FastByIDMap<FastByIDMap<Long>> timestamps) {
    Preconditions.checkArgument(userData != null, "userData is null");

    this.preferenceFromUsers = userData;
    FastByIDMap<Collection<Preference>> prefsForItems = new FastByIDMap<Collection<Preference>>();
    FastIDSet itemIDSet = new FastIDSet();
    int currentCount = 0;
    float maxPrefValue = Float.NEGATIVE_INFINITY;
    float minPrefValue = Float.POSITIVE_INFINITY;
    for (Map.Entry<Long, PreferenceArray> entry : preferenceFromUsers.entrySet()) {
      PreferenceArray prefs = entry.getValue();
      prefs.sortByItem();
      for (Preference preference : prefs) {
        long itemID = preference.getItemID();
        itemIDSet.add(itemID);
        Collection<Preference> prefsForItem = prefsForItems.get(itemID);
        if (prefsForItem == null) {
          prefsForItem = Lists.newArrayListWithCapacity(2);
          prefsForItems.put(itemID, prefsForItem);
        }
        prefsForItem.add(preference);
        float value = preference.getValue();
        if (value > maxPrefValue) {
          maxPrefValue = value;
        }
        if (value < minPrefValue) {
          minPrefValue = value;
        }
      }
      if (++currentCount % 10000 == 0) {
        log.info("Processed {} users", currentCount);
      }
    }
    log.info("Processed {} users", currentCount);

    setMinPreference(minPrefValue);
    setMaxPreference(maxPrefValue);

    this.itemIDs = itemIDSet.toArray();
    itemIDSet = null; // Might help GC -- this is big
    Arrays.sort(itemIDs);

    this.preferenceForItems = toDataMap(prefsForItems, false);

    for (Map.Entry<Long, PreferenceArray> entry : preferenceForItems.entrySet()) {
      entry.getValue().sortByUser();
    }

    this.userIDs = new long[userData.size()];
    int i = 0;
    LongPrimitiveIterator it = userData.keySetIterator();
    while (it.hasNext()) {
      userIDs[i++] = it.next();
    }
    Arrays.sort(userIDs);

    this.timestamps = timestamps;
  }

  /**
   * <p>
   * Creates a new {@link GenericDataModel} containing an immutable copy of the data from another given
   * {@link DataModel}.
   * </p>
   *
   * @param dataModel {@link DataModel} to copy
   * @throws TasteException if an error occurs while retrieving the other {@link DataModel}'s users
   * @deprecated without direct replacement.
   *  Consider {@link #toDataMap(DataModel)} with {@link #GenericDataModel(FastByIDMap)}
   */
  @Deprecated
  public GenericDataModel(DataModel dataModel) throws TasteException {
    this(toDataMap(dataModel));
  }
  
  /**
   * Swaps, in-place, {@link List}s for arrays in {@link Map} values .
   * 
   * @return input value
   */
  public static FastByIDMap<PreferenceArray> toDataMap(FastByIDMap<Collection<Preference>> data,
                                                       boolean byUser) {
    for (Map.Entry<Long,Object> entry : ((FastByIDMap<Object>) (FastByIDMap<?>) data).entrySet()) {
      List<Preference> prefList = (List<Preference>) entry.getValue();
      entry.setValue(byUser ? new GenericUserPreferenceArray(prefList) : new GenericItemPreferenceArray(
          prefList));
    }
    return (FastByIDMap<PreferenceArray>) (FastByIDMap<?>) data;
  }

  /**
   * Exports the simple user IDs and preferences in the data model.
   *
   * @return a {@link FastByIDMap} mapping user IDs to {@link PreferenceArray}s representing
   *  that user's preferences
   */
  public static FastByIDMap<PreferenceArray> toDataMap(DataModel dataModel) throws TasteException {
    FastByIDMap<PreferenceArray> data = new FastByIDMap<PreferenceArray>(dataModel.getNumUsers());
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      long userID = it.nextLong();
      data.put(userID, dataModel.getPreferencesFromUser(userID));
    }
    return data;
  }
  
  /**
   * This is used mostly internally to the framework, and shouldn't be relied upon otherwise.
   */
  public FastByIDMap<PreferenceArray> getRawUserData() {
    return this.preferenceFromUsers;
  }

  /**
   * This is used mostly internally to the framework, and shouldn't be relied upon otherwise.
   */
  public FastByIDMap<PreferenceArray> getRawItemData() {
    return this.preferenceForItems;
  }

  @Override
  public LongPrimitiveArrayIterator getUserIDs() {
    return new LongPrimitiveArrayIterator(userIDs);
  }
  
  /**
   * @throws NoSuchUserException
   *           if there is no such user
   */
  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws NoSuchUserException {
    PreferenceArray prefs = preferenceFromUsers.get(userID);
    if (prefs == null) {
      throw new NoSuchUserException(userID);
    }
    return prefs;
  }
  
  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    PreferenceArray prefs = getPreferencesFromUser(userID);
    int size = prefs.length();
    FastIDSet result = new FastIDSet(size);
    for (int i = 0; i < size; i++) {
      result.add(prefs.getItemID(i));
    }
    return result;
  }
  
  @Override
  public LongPrimitiveArrayIterator getItemIDs() {
    return new LongPrimitiveArrayIterator(itemIDs);
  }
  
  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws NoSuchItemException {
    PreferenceArray prefs = preferenceForItems.get(itemID);
    if (prefs == null) {
      throw new NoSuchItemException(itemID);
    }
    return prefs;
  }
  
  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    PreferenceArray prefs = getPreferencesFromUser(userID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      if (prefs.getItemID(i) == itemID) {
        return prefs.getValue(i);
      }
    }
    return null;
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    if (timestamps == null) {
      return null;
    }
    FastByIDMap<Long> itemTimestamps = timestamps.get(userID);
    if (itemTimestamps == null) {
      throw new NoSuchUserException(userID);
    }
    return itemTimestamps.get(itemID);
  }

  @Override
  public int getNumItems() {
    return itemIDs.length;
  }
  
  @Override
  public int getNumUsers() {
    return userIDs.length;
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) {
    PreferenceArray prefs1 = preferenceForItems.get(itemID);
    return prefs1 == null ? 0 : prefs1.length();
  }
  
  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) {
    PreferenceArray prefs1 = preferenceForItems.get(itemID1);
    if (prefs1 == null) {
      return 0;
    }
    PreferenceArray prefs2 = preferenceForItems.get(itemID2);
    if (prefs2 == null) {
      return 0;
    }

    int size1 = prefs1.length();
    int size2 = prefs2.length();
    int count = 0;
    int i = 0;
    int j = 0;
    long userID1 = prefs1.getUserID(0);
    long userID2 = prefs2.getUserID(0);
    while (true) {
      if (userID1 < userID2) {
        if (++i == size1) {
          break;
        }
        userID1 = prefs1.getUserID(i);
      } else if (userID1 > userID2) {
        if (++j == size2) {
          break;
        }
        userID2 = prefs2.getUserID(j);
      } else {
        count++;
        if (++i == size1 || ++j == size2) {
          break;
        }
        userID1 = prefs1.getUserID(i);
        userID2 = prefs2.getUserID(j);
      }
    }
    return count;
  }

  @Override
  public void removePreference(long userID, long itemID) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void setPreference(long userID, long itemID, float value) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
  // Does nothing
  }

  @Override
  public boolean hasPreferenceValues() {
    return true;
  }
  
  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(200);
    result.append("GenericDataModel[users:");
    for (int i = 0; i < Math.min(3, userIDs.length); i++) {
      if (i > 0) {
        result.append(',');
      }
      result.append(userIDs[i]);
    }
    if (userIDs.length > 3) {
      result.append("...");
    }
    result.append(']');
    return result.toString();
  }
  
}
