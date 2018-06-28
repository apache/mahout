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
 * Like {@link GenericUserPreferenceArray} but stores preferences for one item (all item IDs the same) rather
 * than one user.
 * </p>
 * 
 * @see BooleanItemPreferenceArray
 * @see GenericUserPreferenceArray
 * @see GenericPreference
 */
public final class GenericItemPreferenceArray implements PreferenceArray {

  private static final int USER = 0;
  private static final int VALUE = 2;
  private static final int VALUE_REVERSED = 3;

  private final long[] ids;
  private long id;
  private final float[] values;

  public GenericItemPreferenceArray(int size) {
    this.ids = new long[size];
    values = new float[size];
    this.id = Long.MIN_VALUE; // as a sort of 'unspecified' value
  }

  public GenericItemPreferenceArray(List<? extends Preference> prefs) {
    this(prefs.size());
    int size = prefs.size();
    long itemID = Long.MIN_VALUE;
    for (int i = 0; i < size; i++) {
      Preference pref = prefs.get(i);
      ids[i] = pref.getUserID();
      if (i == 0) {
        itemID = pref.getItemID();
      } else {
        if (itemID != pref.getItemID()) {
          throw new IllegalArgumentException("Not all item IDs are the same");
        }
      }
      values[i] = pref.getValue();
    }
    id = itemID;
  }

  /**
   * This is a private copy constructor for clone().
   */
  private GenericItemPreferenceArray(long[] ids, long id, float[] values) {
    this.ids = ids;
    this.id = id;
    this.values = values;
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
    values[i] = pref.getValue();
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
    return values[i];
  }

  @Override
  public void setValue(int i, float value) {
    values[i] = value;
  }

  @Override
  public void sortByUser() {
    lateralSort(USER);
  }

  @Override
  public void sortByItem() { }

  @Override
  public void sortByValue() {
    lateralSort(VALUE);
  }

  @Override
  public void sortByValueReversed() {
    lateralSort(VALUE_REVERSED);
  }

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

  private void lateralSort(int type) {
    //Comb sort: http://en.wikipedia.org/wiki/Comb_sort
    int length = length();
    int gap = length;
    boolean swapped = false;
    while (gap > 1 || swapped) {
      if (gap > 1) {
        gap /= 1.247330950103979; // = 1 / (1 - 1/e^phi)
      }
      swapped = false;
      int max = length - gap;
      for (int i = 0; i < max; i++) {
        int other = i + gap;
        if (isLess(other, i, type)) {
          swap(i, other);
          swapped = true;
        }
      }
    }
  }

  private boolean isLess(int i, int j, int type) {
    switch (type) {
      case USER:
        return ids[i] < ids[j];
      case VALUE:
        return values[i] < values[j];
      case VALUE_REVERSED:
        return values[i] > values[j];
      default:
        throw new IllegalStateException();
    }
  }

  private void swap(int i, int j) {
    long temp1 = ids[i];
    float temp2 = values[i];
    ids[i] = ids[j];
    values[i] = values[j];
    ids[j] = temp1;
    values[j] = temp2;
  }

  @Override
  public GenericItemPreferenceArray clone() {
    return new GenericItemPreferenceArray(ids.clone(), id, values.clone());
  }

  @Override
  public int hashCode() {
    return (int) (id >> 32) ^ (int) id ^ Arrays.hashCode(ids) ^ Arrays.hashCode(values);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof GenericItemPreferenceArray)) {
      return false;
    }
    GenericItemPreferenceArray otherArray = (GenericItemPreferenceArray) other;
    return id == otherArray.id && Arrays.equals(ids, otherArray.ids) && Arrays.equals(values, otherArray.values);
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
    if (ids == null || ids.length == 0) {
      return "GenericItemPreferenceArray[{}]";
    }
    StringBuilder result = new StringBuilder(20 * ids.length);
    result.append("GenericItemPreferenceArray[itemID:");
    result.append(id);
    result.append(",{");
    for (int i = 0; i < ids.length; i++) {
      if (i > 0) {
        result.append(',');
      }
      result.append(ids[i]);
      result.append('=');
      result.append(values[i]);
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
      return GenericItemPreferenceArray.this.getUserID(i);
    }

    @Override
    public long getItemID() {
      return GenericItemPreferenceArray.this.getItemID(i);
    }

    @Override
    public float getValue() {
      return values[i];
    }

    @Override
    public void setValue(float value) {
      values[i] = value;
    }

  }

}
