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

import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;

/**
 * This implementation maintains three parallel arrays, of {@link User}s, items, and values. The idea is to save
 * allocating {@link Preference} objects themselves. On a 64-bit virtual machine, this should save 12 bytes per element
 * (the overhead of an enclosing {@link Preference} object reference and object header).
 *
 * This is not used yet.
 */
public final class GenericPreferenceArray implements PreferenceArray, Serializable {

  private final User[] users;
  private final Comparable<?>[] itemIDs;
  private final double[] values;

  public GenericPreferenceArray(int size) {
    users = new User[size];
    itemIDs = new Comparable<?>[size];
    values = new double[size];
  }

  @Override
  public Preference get(int i) {
    return new GenericPreference(users[i], itemIDs[i], values[i]);
  }

  @Override
  public void set(int i, Preference pref) {
    users[i] = pref.getUser();
    itemIDs[i] = pref.getItemID();
    values[i] = pref.getValue();
  }

  @Override
  public User getUser(int i) {
    return users[i];
  }

  @Override
  public void setUser(int i, User user) {
    users[i] = user;
  }

  @Override
  public Comparable<?> getItemID(int i) {
    return itemIDs[i];
  }

  @Override
  public void setItemID(int i, Comparable<?> itemID) {
    itemIDs[i] = itemID;
  }


  @Override
  public double getValue(int i) {
    return values[i];
  }

  @Override
  public void setValue(int i, double value) {
    values[i] = value;
  }

}
