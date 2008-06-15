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
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.List;

/**
 * <p>A simple {@link User} which has simply an ID and some {@link Collection} of
 * {@link Preference}s.</p>
 */
public class GenericUser<K extends Comparable<K>> implements User, Serializable {

  private static final Preference[] NO_PREFS = new Preference[0];

  private final K id;
  private final Map<Object, Preference> data;
  // Use an array for maximum performance
  private final Preference[] values;

  public GenericUser(K id, List<Preference> preferences) {
    if (id == null) {
      throw new IllegalArgumentException("id is null");
    }
    this.id = id;
    if (preferences == null || preferences.isEmpty()) {
      data = Collections.emptyMap();
      values = NO_PREFS;
    } else {
      data = new FastMap<Object, Preference>();
      int size = preferences.size();
      values = new Preference[size];
      for (int i = 0; i < size; i++) {
        Preference preference = preferences.get(i);
        values[i] = preference;
        // Is this hacky?
        if (preference instanceof SettableUserPreference) {
          ((SettableUserPreference) preference).setUser(this);
        }
        data.put(preference.getItem().getID(), preference);
      }
      Arrays.sort(values, ByItemPreferenceComparator.getInstance());
    }
  }

  public K getID() {
    return id;
  }

  public Preference getPreferenceFor(Object itemID) {
    return data.get(itemID);
  }

  public Iterable<Preference> getPreferences() {
    return new ArrayIterator<Preference>(values);
  }

  public Preference[] getPreferencesAsArray() {
    return values;
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
    return "User[id:" + String.valueOf(id) + ']';
  }

  public int compareTo(User o) {
    return id.compareTo((K) o.getID());
  }

}
