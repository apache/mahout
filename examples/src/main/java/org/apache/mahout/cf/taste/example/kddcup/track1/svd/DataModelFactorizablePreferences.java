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

package org.apache.mahout.cf.taste.example.kddcup.track1.svd;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;

import java.util.List;

/**
 * can be used to drop {@link DataModel}s into {@link ParallelArraysSGDFactorizer}
 */
public class DataModelFactorizablePreferences implements FactorizablePreferences {

  private final FastIDSet userIDs;
  private final FastIDSet itemIDs;

  private final List<Preference> preferences;

  private final float minPreference;
  private final float maxPreference;

  public DataModelFactorizablePreferences(DataModel dataModel) {

    minPreference = dataModel.getMinPreference();
    maxPreference = dataModel.getMaxPreference();

    try {
      userIDs = new FastIDSet(dataModel.getNumUsers());
      itemIDs = new FastIDSet(dataModel.getNumItems());
      preferences = Lists.newArrayList();

      LongPrimitiveIterator userIDsIterator = dataModel.getUserIDs();
      while (userIDsIterator.hasNext()) {
        long userID = userIDsIterator.nextLong();
        userIDs.add(userID);
        for (Preference preference : dataModel.getPreferencesFromUser(userID)) {
          itemIDs.add(preference.getItemID());
          preferences.add(new GenericPreference(userID, preference.getItemID(), preference.getValue()));
        }
      }
    } catch (TasteException te) {
      throw new IllegalStateException("Unable to create factorizable preferences!", te);
    }
  }

  @Override
  public LongPrimitiveIterator getUserIDs() {
    return userIDs.iterator();
  }

  @Override
  public LongPrimitiveIterator getItemIDs() {
    return itemIDs.iterator();
  }

  @Override
  public Iterable<Preference> getPreferences() {
    return preferences;
  }

  @Override
  public float getMinPreference() {
    return minPreference;
  }

  @Override
  public float getMaxPreference() {
    return maxPreference;
  }

  @Override
  public int numUsers() {
    return userIDs.size();
  }

  @Override
  public int numItems() {
    return itemIDs.size();
  }

  @Override
  public int numPreferences() {
    return preferences.size();
  }
}

