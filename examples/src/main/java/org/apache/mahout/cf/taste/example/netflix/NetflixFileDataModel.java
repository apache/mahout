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

package org.apache.mahout.cf.taste.example.netflix;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericItemPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;

public final class NetflixFileDataModel implements DataModel {

  private final File dataDirectory;

  public NetflixFileDataModel(File dataDirectory) throws IOException {
		if (dataDirectory == null) {
			throw new IllegalArgumentException("dataDirectory is null");
		}
		if (!dataDirectory.exists() || !dataDirectory.isDirectory()) {
			throw new FileNotFoundException(dataDirectory.toString());
		}
    this.dataDirectory = dataDirectory;
  }

  @Override
  public LongPrimitiveIterator getUserIDs() {
    throw new UnsupportedOperationException(); // TODO
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long id) {
    throw new UnsupportedOperationException(); // TODO
  }

  @Override
  public LongPrimitiveIterator getItemIDs() {
    return new MovieIDIterator();
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) {
    throw new UnsupportedOperationException(); // TODO
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) {
    throw new UnsupportedOperationException(); // TODO
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) {
    StringBuilder itemIDPadded = new StringBuilder(5);
    itemIDPadded.append(itemID);
    while (itemIDPadded.length() < 5) {
      itemIDPadded.insert(0, '0');
    }
    List<Preference> prefs = new ArrayList<Preference>();
    File movieFile = new File(new File(dataDirectory, "training_set"), "mv_00" + itemIDPadded + ".txt");
    for (String line : new FileLineIterable(movieFile, true)) {
      int firstComma = line.indexOf((int) ',');
      Integer userID = Integer.valueOf(line.substring(0, firstComma));
      int secondComma = line.indexOf((int) ',', firstComma + 1);
      float rating = Float.parseFloat(line.substring(firstComma + 1, secondComma));
      prefs.add(new GenericPreference(userID, itemID, rating));
    }
    return new GenericItemPreferenceArray(prefs);
  }

  @Override
  public int getNumItems() {
    return MovieIDIterator.COUNT;
  }

  @Override
  public int getNumUsers() {
    throw new UnsupportedOperationException(); // TODO
  }

  @Override
  public int getNumUsersWithPreferenceFor(long... itemIDs) {
    throw new UnsupportedOperationException(); // TODO
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void setPreference(long userID, long itemID, float value) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void removePreference(long userID, long itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

  @Override
  public String toString() {
    return "NetflixFileDataModel";
  }

  private static final class MovieIDIterator implements LongPrimitiveIterator {

    private int next = 1;
    static final int COUNT = 17770;

    @Override
    public long nextLong() {
      if (next <= COUNT) {
        return next++;
      }
      throw new NoSuchElementException();
    }

    @Override
    public long peek() {
      if (next <= COUNT) {
        return next;
      }
      throw new NoSuchElementException();
    }

    @Override
    public boolean hasNext() {
      return next <= COUNT;
    }

    @Override
    public Long next() {
      return nextLong();
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

  }

}
