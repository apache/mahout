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
 * A simple recommender that always estimates preference for an item to be the average of all known preference
 * values for that item. No information about users is taken into account. This implementation is provided for
 * experimentation; while simple and fast, it may not produce very good recommendations.
 * </p>
 */
public final class ItemAverageRecommender extends AbstractRecommender {
  
  private static final Logger log = LoggerFactory.getLogger(ItemAverageRecommender.class);
  
  private final FastByIDMap<RunningAverage> itemAverages;
  private final ReadWriteLock buildAveragesLock;
  private final RefreshHelper refreshHelper;
  
  public ItemAverageRecommender(DataModel dataModel) throws TasteException {
    super(dataModel);
    this.itemAverages = new FastByIDMap<RunningAverage>();
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

    TopItems.Estimator<Long> estimator = new Estimator();

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
    return doEstimatePreference(itemID);
  }
  
  private float doEstimatePreference(long itemID) {
    buildAveragesLock.readLock().lock();
    try {
      RunningAverage average = itemAverages.get(itemID);
      return average == null ? Float.NaN : (float) average.getAverage();
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
        PreferenceArray prefs = dataModel.getPreferencesFromUser(it.nextLong());
        int size = prefs.length();
        for (int i = 0; i < size; i++) {
          long itemID = prefs.getItemID(i);
          RunningAverage average = itemAverages.get(itemID);
          if (average == null) {
            average = new FullRunningAverage();
            itemAverages.put(itemID, average);
          }
          average.addDatum(prefs.getValue(i));
        }
      }
    } finally {
      buildAveragesLock.writeLock().unlock();
    }
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
  public void removePreference(long userID, long itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    Float oldPref = dataModel.getPreferenceValue(userID, itemID);
    super.removePreference(userID, itemID);
    if (oldPref != null) {
      try {
        buildAveragesLock.writeLock().lock();
        RunningAverage average = itemAverages.get(itemID);
        if (average == null) {
          throw new IllegalStateException("No preferences exist for item ID: " + itemID);
        } else {
          average.removeDatum(oldPref);
        }
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
    return "ItemAverageRecommender";
  }
  
  private final class Estimator implements TopItems.Estimator<Long> {
    
    @Override
    public double estimate(Long itemID) {
      return doEstimatePreference(itemID);
    }
  }
  
}
