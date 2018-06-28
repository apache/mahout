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
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.CountingIterator;

/**
 * <p>
 * Like {@link BooleanUserPreferenceArray} but stores preferences for one item (all item IDs the same) rather
 * than one user.
 * </p>
 * 
 * @see BooleanPreference
 * @see BooleanUserPreferenceArray
 * @see GenericItemPreferenceArray
 */
public final class BooleanItemPreferenceArray implements PreferenceArray {
  
  private final long[] ids;
  private long id;
  
  public BooleanItemPreferenceArray(int size) {
    this.ids = new long[size];
    this.id = Long.MIN_VALUE; // as a sort of 'unspecified' value
  }
  
  public BooleanItemPreferenceArray(List<? extends Preference> prefs, boolean forOneUser) {
    this(prefs.size());
    int size = prefs.size();
    for (int i = 0; i < size; i++) {
      Preference pref = prefs.get(i);
      ids[i] = forOneUser ? pref.getItemID() : pref.getUserID();
    }
    if (size > 0) {
      id = forOneUser ? prefs.get(0).getUserID() : prefs.get(0).getItemID();
    }
  }
  
  /**
   * This is a private copy constructor for clone().
   */
  private BooleanItemPreferenceArray(long[] ids, long id) {
    this.ids = ids;
    this.id = id;
  }
  
  @Override
  public int length() {
    return ids.length;
  }
  
  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }
  
  @Override
  public void set(int i, Preference pref) {
    id = pref.getItemID();
    ids[i] = pref.getUserID();
  }
  
  @Override
  public long getUserID(int i) {
    return ids[i];
  }
  
  @Override
  public void setUserID(int i, long userID) {
    ids[i] = userID;
  }
  
  @Override
  public long getItemID(int i) {
    return id;
  }
  
  /**
   * {@inheritDoc}
   * 
   * Note that this method will actually set the item ID for <em>all</em> preferences.
   */
  @Override
  public void setItemID(int i, long itemID) {
    id = itemID;
  }

  /**
   * @return all user IDs
   */
  @Override
  public long[] getIDs() {
    return ids;
  }
  
  @Override
  public float getValue(int i) {
    return 1.0f;
  }
  
  @Override
  public void setValue(int i, float value) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void sortByUser() {
    Arrays.sort(ids);
  }
  
  @Override
  public void sortByItem() { }
  
  @Override
  public void sortByValue() { }
  
  @Override
  public void sortByValueReversed() { }
  
  @Override
  public boolean hasPrefWithUserID(long userID) {
    for (long id : ids) {
      if (userID == id) {
        return true;
      }
    }
    return false;
  }
  
  @Override
  public boolean hasPrefWithItemID(long itemID) {
    return id == itemID;
  }
  
  @Override
  public BooleanItemPreferenceArray clone() {
    return new BooleanItemPreferenceArray(ids.clone(), id);
  }

  @Override
  public int hashCode() {
    return (int) (id >> 32) ^ (int) id ^ Arrays.hashCode(ids);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof BooleanItemPreferenceArray)) {
      return false;
    }
    BooleanItemPreferenceArray otherArray = (BooleanItemPreferenceArray) other;
    return id == otherArray.id && Arrays.equals(ids, otherArray.ids);
  }
  
  @Override
  public Iterator<Preference> iterator() {
    return Iterators.transform(new CountingIterator(length()),
        new Function<Integer, Preference>() {
        @Override
        public Preference apply(Integer from) {
          return new PreferenceView(from);
        }
      });
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(10 * ids.length);
    result.append("BooleanItemPreferenceArray[itemID:");
    result.append(id);
    result.append(",{");
    for (int i = 0; i < ids.length; i++) {
      if (i > 0) {
        result.append(',');
      }
      result.append(ids[i]);
    }
    result.append("}]");
    return result.toString();
  }
  
  private final class PreferenceView implements Preference {
    
    private final int i;
    
    private PreferenceView(int i) {
      this.i = i;
    }
    
    @Override
    public long getUserID() {
      return BooleanItemPreferenceArray.this.getUserID(i);
    }
    
    @Override
    public long getItemID() {
      return BooleanItemPreferenceArray.this.getItemID(i);
    }
    
    @Override
    public float getValue() {
      return 1.0f;
    }
    
    @Override
    public void setValue(float value) {
      throw new UnsupportedOperationException();
    }
    
  }
  
}
