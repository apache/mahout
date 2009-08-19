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

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * <p>Like {@link GenericUserPreferenceArray} but stores, conceptually, {@link BooleanPreference} objects
 * which have no associated preference value.</p>
 * 
 * @see BooleanPreference
 * @see BooleanItemPreferenceArray
 * @see GenericUserPreferenceArray
 */
public final class BooleanUserPreferenceArray implements PreferenceArray {

  private final long[] IDs;
  private long id;

  public BooleanUserPreferenceArray(int size) {
    if (size < 1) {
      throw new IllegalArgumentException("size is less than 1");
    }
    this.IDs = new long[size];
  }

  @Override
  public int length() {
    return IDs.length;
  }

  public BooleanUserPreferenceArray(List<Preference> prefs) {
    this(prefs.size());
    for (int i = 0; i < prefs.size(); i++) {
      Preference pref = prefs.get(i);
      IDs[i] = pref.getItemID();
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
    return 1.0f;
  }

  @Override
  public void setValue(int i, float value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void sortByUser() {
  }

  @Override
  public void sortByItem() {
    Arrays.sort(IDs);
  }

  @Override
  public void sortByValue() {
  }

  @Override
  public void sortByValueReversed() {
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
      return BooleanUserPreferenceArray.this.getUserID(i);
    }

    @Override
    public long getItemID() {
      return BooleanUserPreferenceArray.this.getItemID(i);
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