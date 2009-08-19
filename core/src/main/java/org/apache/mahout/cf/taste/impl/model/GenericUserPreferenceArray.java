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

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * <p>This implementation maintains two parallel arrays, of user IDs and values. The idea is to save
 * allocating {@link Preference} objects themselves. This saves the overhead of {@link Preference} objects
 * but also duplicating the user ID value.</p>
 *
 * @see BooleanUserPreferenceArray
 * @see GenericItemPreferenceArray
 * @see GenericPreference
 */
public final class GenericUserPreferenceArray implements PreferenceArray {

  private static final int ITEM = 1;
  private static final int VALUE = 2;
  private static final int VALUE_REVERSED = 3;

  private final long[] IDs;
  private long id;
  private final float[] values;

  public GenericUserPreferenceArray(int size) {
    if (size < 1) {
      throw new IllegalArgumentException("size is less than 1");
    }
    this.IDs = new long[size];
    values = new float[size];
  }

  @Override
  public int length() {
    return IDs.length;
  }

  public GenericUserPreferenceArray(List<Preference> prefs) {
    this(prefs.size());
    for (int i = 0; i < prefs.size(); i++) {
      Preference pref = prefs.get(i);
      IDs[i] = pref.getItemID();
      values[i] = pref.getValue();
    }
    id = prefs.get(0).getUserID();
  }

  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }

  @Override
  public void set(int i, Preference pref) {
    id = pref.getUserID();
    IDs[i] = pref.getItemID();
    values[i] = pref.getValue();
  }

  @Override
  public long getUserID(int i) {
    return id;
  }

  @Override
  public void setUserID(int i, long userID) {
    id = userID;
  }

  @Override
  public long getItemID(int i) {
    return IDs[i];
  }

  @Override
  public void setItemID(int i, long itemID) {
    IDs[i] = itemID;
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
  }

  @Override
  public void sortByItem() {
    selectionSort(ITEM);
  }

  @Override
  public void sortByValue() {
    selectionSort(VALUE);
  }

  @Override
  public void sortByValueReversed() {
    selectionSort(VALUE_REVERSED);
  }

  private void selectionSort(int type) {
    // I think this sort will prove to be too dumb, but, it's in place and OK for tiny, mostly sorted data
    int max = length();
    for (int i = 0; i < max; i++) {
      int min = i;
      for (int j = i + 1; j < max; j++) {
        if (isLess(j, min, type)) {
          min = j;
        }
      }
      if (i != min) {
        swap(i, min);
      }
    }
  }

  private boolean isLess(int i, int j, int type) {
    switch (type) {
      case ITEM:
        return IDs[i] < IDs[j];
      case VALUE:
        return values[i] < values[j];
      case VALUE_REVERSED:
        return values[i] >= values[j];
      default:
        throw new IllegalStateException();
    }
  }

  private void swap(int i, int j) {
    long temp1 = IDs[i];
    float temp2 = values[i];
    IDs[i] = IDs[j];
    values[i] = values[j];
    IDs[j] = temp1;
    values[j] = temp2;
  }

  @Override
  public GenericUserPreferenceArray clone() {
    try {
      return (GenericUserPreferenceArray) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new AssertionError();
    }
  }

  @Override
  public Iterator<Preference> iterator() {
    return new PreferenceArrayIterator();
  }

  private final class PreferenceArrayIterator implements Iterator<Preference> {
    private int i = 0;
    @Override
    public boolean hasNext() {
      return i < length();
    }
    @Override
    public Preference next() {
      if (i >= length()) {
        throw new NoSuchElementException();
      }
      return new PreferenceView(i++);
    }
    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  private final class PreferenceView implements Preference {

    private final int i;

    private PreferenceView(int i) {
      this.i = i;
    }

    @Override
    public long getUserID() {
      return GenericUserPreferenceArray.this.getUserID(i);
    }

    @Override
    public long getItemID() {
      return GenericUserPreferenceArray.this.getItemID(i);
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