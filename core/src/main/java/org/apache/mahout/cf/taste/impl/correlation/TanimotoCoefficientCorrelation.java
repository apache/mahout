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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.correlation.PreferenceInferrer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>An implementation of a "correlation" based on the
 * <a href="http://en.wikipedia.org/wiki/Jaccard_index">Tanimoto coefficient</a>, or extended
 * Jaccard coefficient.</p>
 *
 * <p>This is intended for "binary" data sets where a user either expersses a generic "yes" preference
 * for an item or has no preference. The actual preference values do not matter here, only their presence
 * or absence.</p>
 *
 * <p>The value returned is in [0,1].</p>
 */
public final class TanimotoCoefficientCorrelation implements UserCorrelation, ItemCorrelation {

  private static final Logger log = LoggerFactory.getLogger(TanimotoCoefficientCorrelation.class);

  private final DataModel dataModel;

  public TanimotoCoefficientCorrelation(DataModel dataModel) {
    this.dataModel = dataModel;
  }

  /**
   * @throws UnsupportedOperationException
   */
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }

  public final double userCorrelation(User user1, User user2) throws TasteException {

    if (user1 == null || user2 == null) {
      throw new IllegalArgumentException("user1 or user2 is null");
    }

    Preference[] xPrefs = user1.getPreferencesAsArray();
    Preference[] yPrefs = user2.getPreferencesAsArray();

    if (xPrefs.length == 0 && yPrefs.length == 0) {
      return Double.NaN;
    }
    if (xPrefs.length == 0 || yPrefs.length == 0) {
      return 0.0;
    }

    Preference xPref = xPrefs[0];
    Preference yPref = yPrefs[0];
    Item xIndex = xPref.getItem();
    Item yIndex = yPref.getItem();
    int xPrefIndex = 1;
    int yPrefIndex = 1;

    int intersectionSize = 0;
    while (true) {
      int compare = xIndex.compareTo(yIndex);
      if (compare == 0) {
        intersectionSize++;
      }
      if (compare <= 0) {
        if (xPrefIndex == xPrefs.length) {
          break;
        }
        xPref = xPrefs[xPrefIndex++];
        xIndex = xPref.getItem();
      }
      if (compare >= 0) {
        if (yPrefIndex == yPrefs.length) {
          break;
        }
        yPref = yPrefs[yPrefIndex++];
        yIndex = yPref.getItem();
      }
    }

    int unionSize = xPrefs.length + yPrefs.length - intersectionSize;

    double result = (double) intersectionSize / (double) unionSize;

    if (log.isTraceEnabled()) {
      log.trace("User correlation between " + user1 + " and " + user2 + " is " + result);
    }
    return result;
  }

  public final double itemCorrelation(Item item1, Item item2) throws TasteException {

    if (item1 == null || item2 == null) {
      throw new IllegalArgumentException("item1 or item2 is null");
    }

    Preference[] xPrefs = dataModel.getPreferencesForItemAsArray(item1.getID());
    Preference[] yPrefs = dataModel.getPreferencesForItemAsArray(item2.getID());

    if (xPrefs.length == 0 && yPrefs.length == 0) {
      return Double.NaN;
    }
    if (xPrefs.length == 0 || yPrefs.length == 0) {
      return 0.0;
    }

    Preference xPref = xPrefs[0];
    Preference yPref = yPrefs[0];
    User xIndex = xPref.getUser();
    User yIndex = yPref.getUser();
    int xPrefIndex = 1;
    int yPrefIndex = 1;

    int intersectionSize = 0;
    while (true) {
      int compare = xIndex.compareTo(yIndex);
      if (compare == 0) {
        intersectionSize++;
      }
      if (compare <= 0) {
        if (xPrefIndex == xPrefs.length) {
          break;
        }
        xPref = xPrefs[xPrefIndex++];
        xIndex = xPref.getUser();
      }
      if (compare >= 0) {
        if (yPrefIndex == yPrefs.length) {
          break;
        }
        yPref = yPrefs[yPrefIndex++];
        yIndex = yPref.getUser();
      }
    }

    int unionSize = xPrefs.length + yPrefs.length - intersectionSize;

    double result = (double) intersectionSize / (double) unionSize;

    if (log.isTraceEnabled()) {
      log.trace("Item correlation between " + item1 + " and " + item2 + " is " + result);
    }
    return result;
  }

  public void refresh() {
    dataModel.refresh();
  }

  @Override
  public final String toString() {
    return "TanimotoCoefficientCorrelation[dataModel:" + dataModel + ']';
  }

}