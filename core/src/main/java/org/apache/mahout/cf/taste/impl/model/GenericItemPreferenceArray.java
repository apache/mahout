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
 * <p>Like {@link GenericUserPreferenceArray} but stores preferences for one item (all item IDs the same)
 * rather than one user.</p>
 *
 * @see BooleanItemPreferenceArray
 * @see GenericUserPreferenceArray
 * @see GenericPreference
 */
public final class GenericItemPreferenceArray implements PreferenceArray {

  private static final int USER = 0;
  private static final int VALUE = 2;
  private static final int VALUE_REVERSED = 3;

  private final long[] IDs;
  private long id;
  private final float[] values;

  public GenericItemPreferenceArray(int size) {
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

  public GenericItemPreferenceArray(List<Preference> prefs) {
    this(prefs.size());
    for (int i = 0; i < prefs.size(); i++) {
      Preference pref = prefs.get(i);
      IDs[i] = pref.getUserID();
      values[i] = pref.getValue();
    }
    id = prefs.get(0).getItemID();
  }

  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }

  @Override
  public void set(int i, Preference pref) {
    id = pref.getItemID();
    IDs[i] = pref.getUserID();
    values[i] = pref.getValue();
  }

  @Override
  public long getUserID(int i) {
    return IDs[i];
  }

  @Override
  public void setUserID(int i, long userID) {
    IDs[i] = userID;
  }

  @Override
  public long getItemID(int i) {
    return id;
  }

  @Override
  public void setItemID(int i, long itemID) {
    id = itemID;
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
    selectionSort(USER);
  }

  @Override
  public void sortByItem() {
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
      case USER:
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
  public GenericItemPreferenceArray clone() {
    try {
      return (GenericItemPreferenceArray) super.clone();
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
