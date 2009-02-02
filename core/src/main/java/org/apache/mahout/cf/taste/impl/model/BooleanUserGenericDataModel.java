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
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A variant on {@link GenericDataModel} which uses the "boolean" classes like {@link BooleanPrefUser}.
 */
public final class BooleanUserGenericDataModel implements DataModel, Serializable {

  private final List<User> users;
  private final Map<Object, User> userMap;
  private final FastSet<Object> itemSet;

  @SuppressWarnings("unchecked")
  public BooleanUserGenericDataModel(Iterable<? extends User> users) {
    if (users == null) {
      throw new IllegalArgumentException("users is null");
    }

    this.userMap = new FastMap<Object, User>();
    this.itemSet = new FastSet<Object>();
    // I'm abusing generics a little here since I want to use this (huge) map to hold Lists,
    // then arrays, and don't want to allocate two Maps at once here.
    for (User user : users) {
      userMap.put(user.getID(), user);
      for (Object itemID : ((BooleanPrefUser<?>) user).getItemIDs()) {
        itemSet.add(itemID);
      }
    }

    List<User> usersCopy = new ArrayList<User>(userMap.values());
    Collections.sort(usersCopy);
    this.users = Collections.unmodifiableList(usersCopy);
  }

  public BooleanUserGenericDataModel(DataModel dataModel) throws TasteException {
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
  public Item getItem(Object id) {
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
    return itemSet.size();
  }

  @Override
  public int getNumUsers() {
    return users.size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(Object... itemIDs) {
    throw new UnsupportedOperationException();

  }

  @Override
  public void setPreference(Object userID, Object itemID, double value) {
    throw new UnsupportedOperationException();
  }

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
    return "BooleanUserGenericDataModel[users:" + users + ']';
  }

}