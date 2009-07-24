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
import org.apache.mahout.cf.taste.impl.common.ArrayIterator;
import org.apache.mahout.cf.taste.impl.common.EmptyIterable;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * <p>A simple {@link DataModel} which uses a given {@link List} of {@link User}s as its data source. This
 * implementation is mostly useful for small experiments and is not recommended for contexts where performance is
 * important.</p>
 */
public final class GenericDataModel implements DataModel, Serializable {

  private static final Logger log = LoggerFactory.getLogger(GenericDataModel.class);

  private static final Preference[] NO_PREFS_ARRAY = new Preference[0];
  private static final Iterable<Preference> NO_PREFS_ITERABLE = new EmptyIterable<Preference>();

  private final List<User> users;
  private final FastMap<Comparable<?>, User> userMap;
  private final List<Comparable<?>> itemIDs;
  private final FastMap<Comparable<?>, Preference[]> preferenceForItems;

  /**
   * <p>Creates a new {@link GenericDataModel} from the given {@link User}s (and their preferences). This {@link
   * DataModel} retains all this information in memory and is effectively immutable.</p>
   *
   * @param users {@link User}s to include in this {@link GenericDataModel}
   */
  @SuppressWarnings("unchecked")
  public GenericDataModel(Iterable<? extends User> users) {
    if (users == null) {
      throw new IllegalArgumentException("users is null");
    }

    this.userMap = new FastMap<Comparable<?>, User>();
    // I'm abusing generics a little here since I want to use this (huge) map to hold Lists,
    // then arrays, and don't want to allocate two Maps at once here.
    FastMap<Comparable<?>, Object> prefsForItems = new FastMap<Comparable<?>, Object>();
    FastSet<Comparable<?>> itemIDSet = new FastSet<Comparable<?>>();
    int currentCount = 0;
    for (User user : users) {
      userMap.put(user.getID(), user);
      Preference[] prefsArray = user.getPreferencesAsArray();
      for (Preference preference : prefsArray) {
        Comparable<?> itemID = preference.getItemID();
        itemIDSet.add(itemID);
        List<Preference> prefsForItem = (List<Preference>) prefsForItems.get(itemID);
        if (prefsForItem == null) {
          prefsForItem = new ArrayList<Preference>();
          prefsForItems.put(itemID, prefsForItem);
        }
        prefsForItem.add(preference);
      }
      currentCount++;
      if (currentCount % 10000 == 0) {
        log.info("Processed {} users", currentCount);
      }
    }
    userMap.rehash();

    this.users = new ArrayList<User>(userMap.values());
    Collections.sort(this.users);

    this.itemIDs = new ArrayList<Comparable<?>>(itemIDSet);
    Collections.sort((List<? extends Comparable>) this.itemIDs);

    prefsForItems.rehash();    
    // Swap out lists for arrays here -- using the same Map. This is why the generics mess is worth it.
    for (Map.Entry<Comparable<?>, Object> entry : prefsForItems.entrySet()) {
      List<Preference> list = (List<Preference>) entry.getValue();
      Preference[] prefsAsArray = list.toArray(new Preference[list.size()]);
      Arrays.sort(prefsAsArray, ByUserPreferenceComparator.getInstance());
      entry.setValue(prefsAsArray);
    }

    // Yeah more generics ugliness
    this.preferenceForItems = (FastMap<Comparable<?>, Preference[]>) (FastMap<Comparable<?>, ?>) prefsForItems;
  }

  /**
   * <p>Creates a new {@link GenericDataModel} containing an immutable copy of the data from another given {@link
   * DataModel}.</p>
   *
   * @param dataModel {@link DataModel} to copy
   * @throws TasteException if an error occurs while retrieving the other {@link DataModel}'s users
   */
  public GenericDataModel(DataModel dataModel) throws TasteException {
    this(dataModel.getUsers());
  }

  @Override
  public Iterable<? extends User> getUsers() {
    return users;
  }

  /** @throws NoSuchUserException if there is no such {@link User} */
  @Override
  public User getUser(Comparable<?> id) throws NoSuchUserException {
    User user = userMap.get(id);
    if (user == null) {
      throw new NoSuchUserException();
    }
    return user;
  }

  @Override
  public Iterable<Comparable<?>> getItemIDs() {
    return itemIDs;
  }

  @Override
  public Iterable<? extends Preference> getPreferencesForItem(Comparable<?> itemID) {
    Preference[] prefs = preferenceForItems.get(itemID);
    return prefs == null ? NO_PREFS_ITERABLE : new ArrayIterator<Preference>(prefs);
  }

  @Override
  public Preference[] getPreferencesForItemAsArray(Comparable<?> itemID) {
    Preference[] prefs = preferenceForItems.get(itemID);
    return prefs == null ? NO_PREFS_ARRAY : prefs;
  }

  @Override
  public int getNumItems() {
    return itemIDs.size();
  }

  @Override
  public int getNumUsers() {
    return users.size();
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
      Preference[] prefs = preferenceForItems.get(itemIDs[0]);
      return prefs == null ? 0 : prefs.length;
    } else {
      Preference[] prefs1 = preferenceForItems.get(itemIDs[0]);
      Preference[] prefs2 = preferenceForItems.get(itemIDs[1]);
      if (prefs1 == null || prefs2 == null) {
        return 0;
      }
      Set<Comparable<?>> users1 = new FastSet<Comparable<?>>(prefs1.length);
      for (Preference aPrefs1 : prefs1) {
        users1.add(aPrefs1.getUser().getID());
      }
      Set<Comparable<?>> users2 = new FastSet<Comparable<?>>(prefs2.length);
      for (Preference aPrefs2 : prefs2) {
        users2.add(aPrefs2.getUser().getID());
      }
      users1.retainAll(users2);
      return users1.size();
    }
  }

  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, double value)
      throws NoSuchUserException, NoSuchItemException {
    getUser(userID).setPreference(itemID, value);
  }

  @Override
  public void removePreference(Comparable<?> userID, Comparable<?> itemID) throws NoSuchUserException {
    getUser(userID).removePreference(itemID);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Does nothing
  }

  @Override
  public String toString() {
    return "GenericDataModel[users:" + (users.size() > 3 ? users.subList(0, 3) + "..." : users) + ']';
  }

}
