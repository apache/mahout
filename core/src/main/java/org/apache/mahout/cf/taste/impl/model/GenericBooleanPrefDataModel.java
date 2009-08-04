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
import org.apache.mahout.cf.taste.model.PreferenceArray;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * <p>A simple {@link DataModel} which uses a given {@link List} of users as its data source. This
 * implementation is mostly useful for small experiments and is not recommended for contexts where performance is
 * important.</p>
 */
public final class GenericBooleanPrefDataModel implements DataModel, Serializable {

  private final List<Comparable<?>> userIDs;
  private final Map<Comparable<?>, FastSet<Comparable<?>>> preferenceFromUsers;
  private final List<Comparable<?>> itemIDs;
  private final Map<Comparable<?>, FastSet<Comparable<?>>> preferenceForItems;

  /**
   * <p>Creates a new {@link GenericDataModel} from the given users (and their preferences). This {@link
   * DataModel} retains all this information in memory and is effectively immutable.</p>
   *
   * @param userData users to include
   */
  @SuppressWarnings("unchecked")
  public GenericBooleanPrefDataModel(Map<Comparable<?>, FastSet<Comparable<?>>> userData) {
    if (userData == null) {
      throw new IllegalArgumentException("userData is null");
    }

    this.preferenceFromUsers = userData;
    this.preferenceForItems = new FastMap<Comparable<?>, FastSet<Comparable<?>>>();
    FastSet<Comparable<?>> itemIDSet = new FastSet<Comparable<?>>();
    for (Map.Entry<Comparable<?>, FastSet<Comparable<?>>> entry : preferenceFromUsers.entrySet()) {
      Comparable<?> userID = entry.getKey();
      FastSet<Comparable<?>> itemIDs = entry.getValue();
      itemIDSet.addAll(itemIDs);
      for (Comparable<?> itemID : itemIDs) {
        FastSet<Comparable<?>> userIDs = preferenceForItems.get(itemID);
        if (userIDs == null) {
          userIDs = new FastSet<Comparable<?>>(2);
          preferenceForItems.put(itemID, userIDs);
        }
        userIDs.add(userID);
      }
    }

    this.itemIDs = new ArrayList<Comparable<?>>(itemIDSet);
    itemIDSet = null;
    Collections.sort((List<? extends Comparable>) this.itemIDs);

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
  public GenericBooleanPrefDataModel(DataModel dataModel) throws TasteException {
    this(toDataMap(dataModel));
  }

  private static Map<Comparable<?>, FastSet<Comparable<?>>> toDataMap(DataModel dataModel) throws TasteException {
    Map<Comparable<?>, FastSet<Comparable<?>>> data = 
            new FastMap<Comparable<?>, FastSet<Comparable<?>>>(dataModel.getNumUsers());
    for (Comparable<?> userID : dataModel.getUserIDs()) {
      data.put(userID, dataModel.getItemIDsFromUser(userID));
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
    FastSet<Comparable<?>> itemIDs = preferenceFromUsers.get(userID);
    if (itemIDs == null) {
      throw new NoSuchUserException();
    }
    PreferenceArray prefArray = new BooleanUserPreferenceArray(itemIDs.size());
    int i = 0;
    for (Comparable<?> itemID : itemIDs) {
      prefArray.setUserID(i, userID);
      prefArray.setItemID(i, itemID);
      i++;
    }
    return prefArray;
  }

  @Override
  public FastSet<Comparable<?>> getItemIDsFromUser(Comparable<?> userID) throws TasteException {
    FastSet<Comparable<?>> itemIDs = preferenceFromUsers.get(userID);
    if (itemIDs == null) {
      throw new NoSuchUserException();
    }
    return itemIDs;
  }

  @Override
  public Iterable<Comparable<?>> getItemIDs() {
    return itemIDs;
  }

  @Override
  public PreferenceArray getPreferencesForItem(Comparable<?> itemID) throws NoSuchItemException {
    FastSet<Comparable<?>> userIDs = preferenceForItems.get(itemID);
    if (userIDs == null) {
      throw new NoSuchItemException();
    }
    PreferenceArray prefArray = new BooleanItemPreferenceArray(userIDs.size());
    int i = 0;
    for (Comparable<?> userID : userIDs) {
      prefArray.setUserID(i, userID);
      prefArray.setItemID(i, itemID);
      i++;
    }
    return prefArray;
  }

  @Override
  public Float getPreferenceValue(Comparable<?> userID, Comparable<?> itemID) throws NoSuchUserException, NoSuchItemException {
    FastSet<Comparable<?>> itemIDs = preferenceFromUsers.get(userID);
    if (itemIDs == null) {
      throw new NoSuchUserException();
    }
    if (itemIDs.contains(itemID)) {
      return 1.0f;
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
  public int getNumUsersWithPreferenceFor(Comparable<?>... itemIDs) throws NoSuchItemException {
    if (itemIDs.length == 0) {
      return 0;
    }
    FastSet<Comparable<?>> intersection = new FastSet<Comparable<?>>();
    FastSet<Comparable<?>> userIDs = preferenceForItems.get(itemIDs[0]);
    if (userIDs == null) {
      throw new NoSuchItemException();
    }
    intersection.addAll(userIDs);
    int i = 1;
    while (!intersection.isEmpty() && i < itemIDs.length) {
      userIDs = preferenceForItems.get(itemIDs[i]);
      if (userIDs == null) {
        throw new NoSuchItemException();
      }
      intersection.retainAll(userIDs);
    }
    return intersection.size();
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
    return "GenericBooleanPrefDataModel[users:" + (userIDs.size() > 3 ? userIDs.subList(0, 3) + "..." : userIDs) + ']';
  }

}
