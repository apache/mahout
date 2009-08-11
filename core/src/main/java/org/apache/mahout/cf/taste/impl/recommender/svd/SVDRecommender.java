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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;

/**
 * <p>A {@link Recommender} which uses Single Value Decomposition to find the main features of the data set.
 * Thanks to Simon Funk for the hints in the implementation.
 */
public final class SVDRecommender extends AbstractRecommender {

  private static final Logger log = LoggerFactory.getLogger(SVDRecommender.class);
  private static final Random random = RandomUtils.getRandom();

  private final RefreshHelper refreshHelper;

  /** Number of features */
  private final int numFeatures;

  private final FastByIDMap<Integer> userMap;
  private FastByIDMap<Integer> itemMap;
  private ExpectationMaximizationSVD emSvd;
  private final List<Preference> cachedPreferences;

  /**
   * @param numFeatures  the number of features
   * @param initialSteps number of initial training steps
   */
  public SVDRecommender(DataModel dataModel, int numFeatures, int initialSteps) throws TasteException {
    this(dataModel, numFeatures);
    train(initialSteps);
  }

  /** @param numFeatures the number of features */
  public SVDRecommender(DataModel dataModel, int numFeatures) throws TasteException {
    super(dataModel);

    this.numFeatures = numFeatures;

    int numUsers = dataModel.getNumUsers();
    userMap = new FastByIDMap<Integer>(numUsers);

    int idx = 0;
    LongPrimitiveIterator userIterator = dataModel.getUserIDs();
    while (userIterator.hasNext()) {
      userMap.put(userIterator.nextLong(), idx++);
    }

    int numItems = dataModel.getNumItems();
    itemMap = new FastByIDMap<Integer>(numItems);

    idx = 0;
    LongPrimitiveIterator itemIterator = dataModel.getItemIDs();
    while (itemIterator.hasNext()) {
      itemMap.put(itemIterator.nextLong(), idx++);
    }

    double average = getAveragePreference();
    double defaultValue = Math.sqrt((average - 1.0) / (double) numFeatures);

    emSvd = new ExpectationMaximizationSVD(numUsers, numItems, numFeatures, defaultValue);
    cachedPreferences = new ArrayList<Preference>(numUsers);
    recachePreferences();

    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        recachePreferences();
        //TODO: train again
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);

  }

  private void recachePreferences() throws TasteException {
    cachedPreferences.clear();
    DataModel dataModel = getDataModel();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      for  (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
        cachedPreferences.add(pref);
      }
    }
  }

  private double getAveragePreference() throws TasteException {
    RunningAverage average = new FullRunningAverage();
    DataModel dataModel = getDataModel();
    LongPrimitiveIterator it = dataModel.getUserIDs();
    while (it.hasNext()) {
      for (Preference pref : dataModel.getPreferencesFromUser(it.nextLong())) {
        average.addDatum(pref.getValue());
      }
    }
    return average.getAverage();
  }

  public void train(int steps) {
    for (int i = 0; i < steps; i++) {
      nextTrainStep();
    }
  }

  private void nextTrainStep() {
    Collections.shuffle(cachedPreferences, random);
    for (int i = 0; i < numFeatures; i++) {
      for (Preference pref : cachedPreferences) {
        int useridx = userMap.get(pref.getUserID());
        int itemidx = itemMap.get(pref.getItemID());
        emSvd.train(useridx, itemidx, i, pref.getValue());
      }
    }
  }

  private float predictRating(int user, int item) {
    return (float) emSvd.getDotProduct(user, item);
  }


  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    Integer useridx = userMap.get(userID);
    if (useridx == null) {
      throw new NoSuchUserException();
    }
    Integer itemidx = itemMap.get(itemID);
    if (itemidx == null) {
      throw new NoSuchItemException();
    }
    return predictRating(useridx, itemidx);
  }

  @Override
  public List<RecommendedItem> recommend(long userID,
                                         int howMany,
                                         Rescorer<Long> rescorer) throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    FastIDSet allItemIDs = getAllOtherItems(userID);

    TopItems.Estimator<Long> estimator = new Estimator(userID);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs.iterator(), rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "SVDRecommender[numFeatures:" + numFeatures + ']';
  }

  private final class Estimator implements TopItems.Estimator<Long> {

    private final long theUserID;

    private Estimator(long theUserID) {
      this.theUserID = theUserID;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return estimatePreference(theUserID, itemID);
    }
  }

}
