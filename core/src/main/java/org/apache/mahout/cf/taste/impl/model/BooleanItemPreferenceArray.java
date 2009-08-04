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

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

/**
 * <p>Like {@link BooleanUserPreferenceArray} but stores preferences for one item (all item IDs the same)
 * rather than one user.</p>
 *
 * @see BooleanPreference
 * @see BooleanUserPreferenceArray
 * @see GenericItemPreferenceArray
 */
public final class BooleanItemPreferenceArray implements PreferenceArray, Serializable {

  private final Comparable<?>[] IDs;
  private Comparable<?> id;

  public BooleanItemPreferenceArray(int size) {
    if (size < 1) {
      throw new IllegalArgumentException("size is less than 1");
    }
    this.IDs = new Comparable<?>[size];
  }

  public int length() {
    return IDs.length;
  }

  public BooleanItemPreferenceArray(List<Preference> prefs, boolean forOneUser) {
    this(prefs.size());
    for (int i = 0; i < prefs.size(); i++) {
      Preference pref = prefs.get(i);
      IDs[i] = forOneUser ? pref.getItemID() : pref.getUserID();
    }
    id = forOneUser ? prefs.get(0).getUserID() : prefs.get(0).getItemID();
  }

  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }

  @Override
  public void set(int i, Preference pref) {
    id = pref.getItemID();
    IDs[i] = pref.getUserID();
  }

  @Override
  public Comparable<?> getUserID(int i) {
    return IDs[i];
  }

  @Override
  public void setUserID(int i, Comparable<?> userID) {
    IDs[i] = userID;
  }

  @Override
  public Comparable<?> getItemID(int i) {
    return id;
  }

  @Override
  public void setItemID(int i, Comparable<?> itemID) {
    id = itemID;
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
    selectionSort();
  }

  @Override
  public void sortByItem() {
  }

  @Override
  public void sortByValue() {
  }

  @Override
  public void sortByValueReversed() {
  }

  private void selectionSort() {
    // I think this sort will prove to be too dumb, but, it's in place and OK for tiny, mostly sorted data
    int max = length();
    for (int i = 0; i < max; i++) {
      int min = i;
      for (int j = i + 1; j < max; j++) {
        if (((Comparable<Object>) IDs[i]).compareTo(IDs[j]) < 0) {
          min = j;
        }
      }
      if (i != min) {
        Comparable<?> temp1 = IDs[i];
        IDs[i] = IDs[min];
        IDs[min] = temp1;
      }
    }
  }

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
    public Comparable<?> getUserID() {
      return BooleanItemPreferenceArray.this.getUserID(i);
    }

    @Override
    public Comparable<?> getItemID() {
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