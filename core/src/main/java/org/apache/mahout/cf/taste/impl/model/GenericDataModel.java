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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.ArrayIterator;
import org.apache.mahout.cf.taste.impl.common.EmptyIterable;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * <p>A simple {@link DataModel} which uses a given {@link List} of {@link User}s as
 * its data source. This implementation is mostly useful for small experiments and is not
 * recommended for contexts where performance is important.</p>
 */
public final class GenericDataModel implements DataModel, Serializable {

  private static final Preference[] NO_PREFS_ARRAY = new Preference[0];
  private static final Iterable<Preference> NO_PREFS_ITERABLE = new EmptyIterable<Preference>();

  private final List<User> users;
  private final Map<Object, User> userMap;
  private final List<Item> items;
  private final Map<Object, Item> itemMap;
  private final Map<Object, Preference[]> preferenceForItems;

  /**
   * <p>Creates a new {@link GenericDataModel} from the given {@link User}s (and their preferences).
   * This {@link DataModel} retains all this information in memory and is effectively immutable.</p>
   *
   * @param users {@link User}s to include in this {@link GenericDataModel}
   */
  @SuppressWarnings("unchecked")
  public GenericDataModel(Iterable<? extends User> users) {
    if (users == null) {
      throw new IllegalArgumentException("users is null");
    }

    this.userMap = new FastMap<Object, User>();
    this.itemMap = new FastMap<Object, Item>();
    // I'm abusing generics a little here since I want to use this (huge) map to hold Lists,
    // then arrays, and don't want to allocate two Maps at once here.
    Map<Object, Object> prefsForItems = new FastMap<Object, Object>();
    for (User user : users) {
      userMap.put(user.getID(), user);
      Preference[] prefsArray = user.getPreferencesAsArray();
      for (Preference preference : prefsArray) {
        Item item = preference.getItem();
        Object itemID = item.getID();
        itemMap.put(itemID, item);
        List<Preference> prefsForItem = (List<Preference>) prefsForItems.get(itemID);
        if (prefsForItem == null) {
          prefsForItem = new ArrayList<Preference>();
          prefsForItems.put(itemID, prefsForItem);
        }
        prefsForItem.add(preference);
      }
    }

    List<User> usersCopy = new ArrayList<User>(userMap.values());
    Collections.sort(usersCopy);
    this.users = Collections.unmodifiableList(usersCopy);

    List<Item> itemsCopy = new ArrayList<Item>(itemMap.values());
    Collections.sort(itemsCopy);
    this.items = Collections.unmodifiableList(itemsCopy);

    // Swap out lists for arrays here -- using the same Map. This is why the generics mess is worth it.
    for (Map.Entry<Object, Object> entry : prefsForItems.entrySet()) {
      List<Preference> list = (List<Preference>) entry.getValue();
      Preference[] prefsAsArray = list.toArray(new Preference[list.size()]);
      Arrays.sort(prefsAsArray, ByUserPreferenceComparator.getInstance());
      entry.setValue(prefsAsArray);
    }
    // Yeah more generics ugliness
    this.preferenceForItems = (Map<Object, Preference[]>) (Map<Object, ?>) prefsForItems;
  }

  /**
   * <p>Creates a new {@link GenericDataModel} containing an immutable copy of the data from another
   * given {@link DataModel}.</p>
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

  /**
   * @throws NoSuchElementException if there is no such {@link User}
   */
  @Override
  public User getUser(Object id) {
    User user = userMap.get(id);
    if (user == null) {
      throw new NoSuchElementException();
    }
    return user;
  }

  @Override
  public Iterable<? extends Item> getItems() {
    return items;
  }

  /**
   * @throws NoSuchElementException if there is no such {@link Item}
   */
  @Override
  public Item getItem(Object id) {
    Item item = itemMap.get(id);
    if (item == null) {
      throw new NoSuchElementException();
    }
    return item;
  }

  @Override
  public Iterable<? extends Preference> getPreferencesForItem(Object itemID) {
    Preference[] prefs = preferenceForItems.get(itemID);
    return prefs == null ? NO_PREFS_ITERABLE : new ArrayIterator<Preference>(prefs);
  }

  @Override
  public Preference[] getPreferencesForItemAsArray(Object itemID) {
    Preference[] prefs = preferenceForItems.get(itemID);
    return prefs == null ? NO_PREFS_ARRAY : prefs;
  }

  @Override
  public int getNumItems() {
    return items.size();
  }

  @Override
  public int getNumUsers() {
    return users.size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(Object... itemIDs) {
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
      Set<Object> users1 = new FastSet<Object>(prefs1.length);
      for (Preference aPrefs1 : prefs1) {
        users1.add(aPrefs1.getUser().getID());
      }
      Set<Object> users2 = new FastSet<Object>(prefs2.length);
      for (Preference aPrefs2 : prefs2) {
        users2.add(aPrefs2.getUser().getID());
      }
      users1.retainAll(users2);
      return users1.size();
    }
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void setPreference(Object userID, Object itemID, double value) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void removePreference(Object userID, Object itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Does nothing
  }

  @Override
  public String toString() {
    return "GenericDataModel[users:" + users + ']';
  }

}
