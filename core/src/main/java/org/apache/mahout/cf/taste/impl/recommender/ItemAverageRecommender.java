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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Collection;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.Callable;

/**
 * <p>A simple recommender that always estimates preference for an {@link Item} to be the average of
 * all known preference values for that {@link Item}. No information about {@link User}s is taken into
 * account. This implementation is provided for experimentation; while simple and fast, it may not
 * produce very good recommendations.</p>
 */
public final class ItemAverageRecommender extends AbstractRecommender {

  private static final Logger log = LoggerFactory.getLogger(ItemAverageRecommender.class);

  private final Map<Object, RunningAverage> itemAverages;
  private boolean averagesBuilt;
  private final ReadWriteLock buildAveragesLock;
  private final RefreshHelper refreshHelper;

  public ItemAverageRecommender(DataModel dataModel) {
    super(dataModel);
    this.itemAverages = new FastMap<Object, RunningAverage>();
    this.buildAveragesLock = new ReentrantReadWriteLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      public Object call() throws TasteException {
        buildAverageDiffs();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
  }

  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
          throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }
    if (rescorer == null) {
      throw new IllegalArgumentException("rescorer is null");
    }
    log.debug("Recommending items for user ID '{}'", userID);
    checkAverageDiffsBuilt();

    User theUser = getDataModel().getUser(userID);
    Set<Item> allItems = getAllOtherItems(theUser);

    TopItems.Estimator<Item> estimator = new Estimator();

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItems, rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  public double estimatePreference(Object userID, Object itemID) throws TasteException {
    DataModel model = getDataModel();
    User theUser = model.getUser(userID);
    Preference actualPref = theUser.getPreferenceFor(itemID);
    if (actualPref != null) {
      return actualPref.getValue();
    }
    checkAverageDiffsBuilt();
    return doEstimatePreference(itemID);
  }

  private double doEstimatePreference(Object itemID) {
    buildAveragesLock.readLock().lock();
    try {
      RunningAverage average = itemAverages.get(itemID);
      return average == null ? Double.NaN : average.getAverage();
    } finally {
      buildAveragesLock.readLock().unlock();
    }
  }

  private void checkAverageDiffsBuilt() throws TasteException {
    if (!averagesBuilt) {
      buildAverageDiffs();
    }
  }

  private void buildAverageDiffs() throws TasteException {
    try {
      buildAveragesLock.writeLock().lock();
      DataModel dataModel = getDataModel();
      for (User user : dataModel.getUsers()) {
        Preference[] prefs = user.getPreferencesAsArray();
        for (int i = 0; i < prefs.length; i++) {
          Preference pref = prefs[i];
          Object itemID = pref.getItem().getID();
          RunningAverage average = itemAverages.get(itemID);
          if (average == null) {
            average = new FullRunningAverage();
            itemAverages.put(itemID, average);
          }
          average.addDatum(pref.getValue());
        }
      }
      averagesBuilt = true;
    } finally {
      buildAveragesLock.writeLock().unlock();
    }
  }

  @Override
  public void setPreference(Object userID, Object itemID, double value) throws TasteException {
    DataModel dataModel = getDataModel();
    double prefDelta;
    try {
      User theUser = dataModel.getUser(userID);
      Preference oldPref = theUser.getPreferenceFor(itemID);
      prefDelta = oldPref == null ? value : value - oldPref.getValue();
    } catch (NoSuchElementException nsee) {
      prefDelta = value;
    }
    super.setPreference(userID, itemID, value);
    try {
      buildAveragesLock.writeLock().lock();
      RunningAverage average = itemAverages.get(itemID);
      if (average == null) {
        RunningAverage newAverage = new FullRunningAverage();
        newAverage.addDatum(prefDelta);
        itemAverages.put(itemID, newAverage);
      } else {
        average.changeDatum(prefDelta);
      }
    } finally {
      buildAveragesLock.writeLock().unlock();
    }
  }

  @Override
  public void removePreference(Object userID, Object itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    User theUser = dataModel.getUser(userID);
    Preference oldPref = theUser.getPreferenceFor(itemID);
    super.removePreference(userID, itemID);
    if (oldPref != null) {
      try {
        buildAveragesLock.writeLock().lock();
        RunningAverage average = itemAverages.get(itemID);
        if (average == null) {
          throw new IllegalStateException("No preferences exist for item ID: " + itemID);
        } else {
          average.removeDatum(oldPref.getValue());
        }
      } finally {
        buildAveragesLock.writeLock().unlock();
      }
    }
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "ItemAverageRecommender";
  }

  private final class Estimator implements TopItems.Estimator<Item> {

    public double estimate(Item item) {
      return doEstimatePreference(item.getID());
    }
  }

}
