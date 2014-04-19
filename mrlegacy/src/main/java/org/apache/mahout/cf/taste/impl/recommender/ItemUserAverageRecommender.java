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

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Like {@link ItemAverageRecommender}, except that estimated preferences are adjusted for the users' average
 * preference value. For example, say user X has not rated item Y. Item Y's average preference value is 3.5.
 * User X's average preference value is 4.2, and the average over all preference values is 4.0. User X prefers
 * items 0.2 higher on average, so, the estimated preference for user X, item Y is 3.5 + 0.2 = 3.7.
 * </p>
 */
public final class ItemUserAverageRecommender extends AbstractRecommender {
  
  private static final Logger log = LoggerFactory.getLogger(ItemUserAverageRecommender.class);
  
  private final FastByIDMap<RunningAverage> itemAverages;
  private final FastByIDMap<RunningAverage> userAverages;
  private final RunningAverage overallAveragePrefValue;
  private final ReadWriteLock buildAveragesLock;
  private final RefreshHelper refreshHelper;
  
  public ItemUserAverageRecommender(DataModel dataModel) throws TasteException {
    super(dataModel);
    this.itemAverages = new FastByIDMap<RunningAverage>();
    this.userAverages = new FastByIDMap<RunningAverage>();
    this.overallAveragePrefValue = new FullRunningAverage();
    this.buildAveragesLock = new ReentrantReadWriteLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildAverageDiffs();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    buildAverageDiffs();
  }
  
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);

    PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
    FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser);

    TopItems.Estimator<Long> estimator = new Estimator(userID);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
      estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }
  
  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    Float actualPref = dataModel.getPreferenceValue(userID, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    return doEstimatePreference(userID, itemID);
  }
  
  private float doEstimatePreference(long userID, long itemID) {
    buildAveragesLock.readLock().lock();
    try {
      RunningAverage itemAverage = itemAverages.get(itemID);
      if (itemAverage == null) {
        return Float.NaN;
      }
      RunningAverage userAverage = userAverages.get(userID);
      if (userAverage == null) {
        return Float.NaN;
      }
      double userDiff = userAverage.getAverage() - overallAveragePrefValue.getAverage();
      return (float) (itemAverage.getAverage() + userDiff);
    } finally {
      buildAveragesLock.readLock().unlock();
    }
  }

  private void buildAverageDiffs() throws TasteException {
    try {
      buildAveragesLock.writeLock().lock();
      DataModel dataModel = getDataModel();
      LongPrimitiveIterator it = dataModel.getUserIDs();
      while (it.hasNext()) {
        long userID = it.nextLong();
        PreferenceArray prefs = dataModel.getPreferencesFromUser(userID);
        int size = prefs.length();
        for (int i = 0; i < size; i++) {
          long itemID = prefs.getItemID(i);
          float value = prefs.getValue(i);
          addDatumAndCreateIfNeeded(itemID, value, itemAverages);
          addDatumAndCreateIfNeeded(userID, value, userAverages);
          overallAveragePrefValue.addDatum(value);
        }
      }
    } finally {
      buildAveragesLock.writeLock().unlock();
    }
  }
  
  private static void addDatumAndCreateIfNeeded(long itemID, float value, FastByIDMap<RunningAverage> averages) {
    RunningAverage itemAverage = averages.get(itemID);
    if (itemAverage == null) {
      itemAverage = new FullRunningAverage();
      averages.put(itemID, itemAverage);
    }
    itemAverage.addDatum(value);
  }
  
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    DataModel dataModel = getDataModel();
    double prefDelta;
    try {
      Float oldPref = dataModel.getPreferenceValue(userID, itemID);
      prefDelta = oldPref == null ? value : value - oldPref;
    } catch (NoSuchUserException nsee) {
      prefDelta = value;
    }
    super.setPreference(userID, itemID, value);
    try {
      buildAveragesLock.writeLock().lock();
      RunningAverage itemAverage = itemAverages.get(itemID);
      if (itemAverage == null) {
        RunningAverage newItemAverage = new FullRunningAverage();
        newItemAverage.addDatum(prefDelta);
        itemAverages.put(itemID, newItemAverage);
      } else {
        itemAverage.changeDatum(prefDelta);
      }
      RunningAverage userAverage = userAverages.get(userID);
      if (userAverage == null) {
        RunningAverage newUserAveragae = new FullRunningAverage();
        newUserAveragae.addDatum(prefDelta);
        userAverages.put(userID, newUserAveragae);
      } else {
        userAverage.changeDatum(prefDelta);
      }
      overallAveragePrefValue.changeDatum(prefDelta);
    } finally {
      buildAveragesLock.writeLock().unlock();
    }
  }
  
  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    Float oldPref = dataModel.getPreferenceValue(userID, itemID);
    super.removePreference(userID, itemID);
    if (oldPref != null) {
      try {
        buildAveragesLock.writeLock().lock();
        RunningAverage itemAverage = itemAverages.get(itemID);
        if (itemAverage == null) {
          throw new IllegalStateException("No preferences exist for item ID: " + itemID);
        }
        itemAverage.removeDatum(oldPref);
        RunningAverage userAverage = userAverages.get(userID);
        if (userAverage == null) {
          throw new IllegalStateException("No preferences exist for user ID: " + userID);
        }
        userAverage.removeDatum(oldPref);
        overallAveragePrefValue.removeDatum(oldPref);
      } finally {
        buildAveragesLock.writeLock().unlock();
      }
    }
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  @Override
  public String toString() {
    return "ItemUserAverageRecommender";
  }
  
  private final class Estimator implements TopItems.Estimator<Long> {
    
    private final long userID;
    
    private Estimator(long userID) {
      this.userID = userID;
    }
    
    @Override
    public double estimate(Long itemID) {
      return doEstimatePreference(userID, itemID);
    }
  }
  
}
