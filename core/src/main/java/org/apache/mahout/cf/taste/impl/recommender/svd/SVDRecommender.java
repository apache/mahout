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
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
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

  private final Map<Comparable<?>, Integer> userMap;
  private Map<Comparable<?>, Integer> itemMap;
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
    userMap = new FastMap<Comparable<?>, Integer>(numUsers);

    int idx = 0;
    for (User user : dataModel.getUsers()) {
      userMap.put(user.getID(), idx++);
    }

    int numItems = dataModel.getNumItems();
    itemMap = new FastMap<Comparable<?>, Integer>(numItems);

    idx = 0;
    for (Comparable<?> itemID : dataModel.getItemIDs()) {
      itemMap.put(itemID, idx++);
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
    for (User user : getDataModel().getUsers()) {
      for (Preference pref : user.getPreferences()) {
        cachedPreferences.add(pref);
      }
    }
  }

  private double getAveragePreference() throws TasteException {
    RunningAverage average = new FullRunningAverage();
    for (User user : getDataModel().getUsers()) {
      for (Preference pref : user.getPreferences()) {
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
        int useridx = userMap.get(pref.getUser().getID());
        int itemidx = itemMap.get(pref.getItemID());
        emSvd.train(useridx, itemidx, i, pref.getValue());
      }
    }
  }

  private double predictRating(int user, int item) {
    return emSvd.getDotProduct(user, item);
  }


  @Override
  public double estimatePreference(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
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
  public List<RecommendedItem> recommend(Comparable<?> userID, 
                                         int howMany,
                                         Rescorer<Comparable<?>> rescorer) throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    User theUser = getDataModel().getUser(userID);

    Set<Comparable<?>> allItemIDs = getAllOtherItems(theUser);

    TopItems.Estimator<Comparable<?>> estimator = new Estimator(theUser);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItemIDs, rescorer, estimator);

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

  private final class Estimator implements TopItems.Estimator<Comparable<?>> {

    private final User theUser;

    private Estimator(User theUser) {
      this.theUser = theUser;
    }

    @Override
    public double estimate(Comparable<?> itemID) throws TasteException {
      return estimatePreference(theUser.getID(), itemID);
    }
  }

}
