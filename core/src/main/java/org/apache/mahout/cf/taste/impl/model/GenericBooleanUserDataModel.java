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

/**
 * TODO many methods here have not been implemented properly yet
 */
public final class GenericBooleanUserDataModel implements DataModel, Serializable {

  private final List<User> users;
  private final Map<Object, User> userMap;
  private final Object[] itemIDs;
  //private final Set<Object> itemIDSet;

  public GenericBooleanUserDataModel(Iterable<? extends User> users) {
    if (users == null) {
      throw new IllegalArgumentException("users is null");
    }
    this.users = new ArrayList<User>();
    this.userMap = new FastMap<Object, User>();
    List<Object> itemIDs = new ArrayList<Object>();
    for (User user : users) {
      if (!(user instanceof BooleanPrefUser)) {
        throw new IllegalArgumentException("Must use a source of BooleanPrefUsers");
      }
      this.users.add(user);
      userMap.put(user.getID(), user);
      itemIDs.addAll(((BooleanPrefUser) user).getItemIDs());
    }
    Collections.sort(this.users);

    this.itemIDs = itemIDs.toArray();
    itemIDs = null;
    Arrays.sort(this.itemIDs);
    //this.itemIDSet = new FastSet<Object>(this.itemIDs.length);
    //itemIDSet.addAll(Arrays.asList(this.itemIDs));
  }

  public GenericBooleanUserDataModel(DataModel dataModel) throws TasteException {
    this(dataModel.getUsers());
  }

  @Override
  public Iterable<? extends User> getUsers() {
    return users;
  }

  @Override
  public User getUser(Object id) throws NoSuchUserException {
    User user = userMap.get(id);
    if (user == null) {
      throw new NoSuchUserException();
    }
    return user;
  }

  @Override
  public Iterable<? extends Item> getItems() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Item getItem(Object id) throws NoSuchItemException {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterable<? extends Preference> getPreferencesForItem(Object itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Preference[] getPreferencesForItemAsArray(Object itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getNumItems() {
    return itemIDs.length;
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
    throw new UnsupportedOperationException();
  }

  @Override
  public void setPreference(Object userID, Object itemID, double value)
      throws NoSuchUserException, NoSuchItemException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void removePreference(Object userID, Object itemID) throws NoSuchUserException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Does nothing
  }

  @Override
  public String toString() {
    return "GenericBooleanUserDataModel[users:" + (users.size() > 3 ? users.subList(0, 3) + "..." : users) + ']';
  }

}