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
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.ArrayIterator;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A variant of {@link GenericUser} which is appropriate when users express only a "yes" preference for
 * an item, or none at all. The preference value for all items is considered to be 1.0.
 */
public class BooleanPrefUser<K extends Comparable<K>> implements User, Serializable {

  private final K id;
  private final FastSet<Object> itemIDs;

  public BooleanPrefUser(K id, FastSet<Object> itemIDs) {
    if (id == null || itemIDs == null || itemIDs.isEmpty()) {
      throw new IllegalArgumentException("id or itemIDs is null or empty");
    }
    this.id = id;
    this.itemIDs = itemIDs;
  }

  @Override
  public K getID() {
    return id;
  }

  @Override
  public Preference getPreferenceFor(Object itemID) {
    return itemIDs.contains(itemID) ? buildPreference(itemID) : null;
  }

  @Override
  public Iterable<Preference> getPreferences() {
    return new ArrayIterator<Preference>(getPreferencesAsArray());
  }

  @Override
  public Preference[] getPreferencesAsArray() {
    Preference[] result = new Preference[itemIDs.size()];
    int i = 0;
    for (Object itemID : itemIDs) {
      result[i] = buildPreference(itemID);
      i++;
    }
    Arrays.sort(result, ByItemPreferenceComparator.getInstance());
    return result;
  }

  private Preference buildPreference(Object itemID) {
    return new BooleanPreference(this, new GenericItem<String>(itemID.toString()));
  }

  /**
   * @return true iff this user expresses a preference for the given item
   */
  public boolean hasPreferenceFor(Object itemID) {
    return itemIDs.contains(itemID);
  }

  /**
   * @return all item IDs the user expresses a preference for
   */
  public FastSet<Object> getItemIDs() {
    return itemIDs;
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof User && ((User) obj).getID().equals(id);
  }

  @Override
  public String toString() {
    return "User[id:" + id + ']';
  }

  @Override
  @SuppressWarnings("unchecked")
  public int compareTo(User o) {
    return id.compareTo((K) o.getID());
  }

}