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

import org.apache.mahout.cf.taste.impl.common.ArrayIterator;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A variant of {@link GenericUser} which is appropriate when users express only a "yes" preference for an item, or none
 * at all. The preference value for all items is considered to be 1.0.
 */
public class BooleanPrefUser implements User, Serializable {

  private final Comparable id;
  private final FastSet<Comparable<?>> itemIDs;

  public BooleanPrefUser(Comparable<?> id, FastSet<Comparable<?>> itemIDs) {
    if (id == null || itemIDs == null || itemIDs.isEmpty()) {
      throw new IllegalArgumentException("id or itemIDs is null or empty");
    }
    this.id = id;
    this.itemIDs = itemIDs;
  }

  @Override
  public Comparable<?> getID() {
    return id;
  }

  @Override
  public Preference getPreferenceFor(Comparable<?> itemID) {
    return itemIDs.contains(itemID) ? buildPreference(itemID) : null;
  }

  /** Note that the value parameter is ignored; it is as if it were always 1.0. */
  @Override
  public void setPreference(Comparable<?> itemID, double value) {
    itemIDs.add(itemID);
  }

  @Override
  public void removePreference(Comparable<?> itemID) {
    itemIDs.remove(itemID);
  }

  @Override
  public Iterable<Preference> getPreferences() {
    return new ArrayIterator<Preference>(getPreferencesAsArray());
  }

  @Override
  public Preference[] getPreferencesAsArray() {
    Preference[] result = new Preference[itemIDs.size()];
    int i = 0;
    for (Comparable<?> itemID : itemIDs) {
      result[i] = buildPreference(itemID);
      i++;
    }
    Arrays.sort(result, ByItemPreferenceComparator.getInstance());
    return result;
  }

  private Preference buildPreference(Comparable<?> itemID) {
    return new BooleanPreference(this, itemID);
  }

  /** @return true iff this user expresses a preference for the given item */
  public boolean hasPreferenceFor(Comparable<?> itemID) {
    return itemIDs.contains(itemID);
  }

  /** @return all item IDs the user expresses a preference for */
  public FastSet<Comparable<?>> getItemIDs() {
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
    return id.compareTo(o.getID());
  }

}