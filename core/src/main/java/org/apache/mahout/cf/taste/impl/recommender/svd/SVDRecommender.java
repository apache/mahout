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
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.uncommons.maths.statistics.DataSet;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * <p>A {@link Recommender} which uses Single Value Decomposition to
 * find the main features of the {@link DataSet}.
 * Thanks to Simon Funk for the hints in the implementation.
 */
public final class SVDRecommender extends AbstractRecommender {

  private static final Logger log = LoggerFactory.getLogger(SVDRecommender.class);

  private final RefreshHelper refreshHelper;

  /** Number of features */
  private final int numFeatures;

  private final Map<Object, Integer> userMap;
  private Map<Object, Integer> itemMap;
  private ExpectationMaximizationSVD emSvd;

  /**
   * @param dataModel
   * @param numFeatures  the number of features
   * @param initialSteps number of initial training steps
   */
  public SVDRecommender(DataModel dataModel, int numFeatures, int initialSteps) throws TasteException{
    this(dataModel, numFeatures);
    train(initialSteps);
  }

  /**
   * @param dataModel
   * @param numFeatures the number of features
   */
  public SVDRecommender(DataModel dataModel, int numFeatures) throws TasteException {
    super(dataModel);

    this.numFeatures = numFeatures;

    int numUsers = dataModel.getNumUsers();
    userMap = new FastMap<Object, Integer>(numUsers);

    int idx = 0;
    for (User user : dataModel.getUsers()) {
      userMap.put(user.getID(), idx++);
    }

    int numItems = dataModel.getNumItems();
    itemMap = new FastMap<Object, Integer>(numItems);

    idx = 0;
    for (Item item : dataModel.getItems()) {
      itemMap.put(item.getID(), idx++);
    }

    double average = getAveragePreference();
    double defaultValue = Math.sqrt((average - 1.0) / numFeatures);

    emSvd = new ExpectationMaximizationSVD(numUsers, numItems, numFeatures, defaultValue);


    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() {
        //TODO: train again
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
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

  public void train(int steps) throws TasteException {
    for (int i = 0; i < steps; i++) {
      nextTrainStep();
    }
  }

  private void nextTrainStep() throws TasteException {
    for (int i = 0; i < numFeatures; i++) {
      for (User user : getDataModel().getUsers()) {
        int useridx = userMap.get(user.getID());
        for (Preference pref : user.getPreferencesAsArray()) {
          int itemidx = itemMap.get(pref.getItem().getID());
          emSvd.train(useridx, itemidx, i, pref.getValue());
        }
      }
    }
  }

  private double predictRating(int user, int item) {
    return emSvd.getDotProduct(user, item);
  }


  @Override
  public double estimatePreference(Object userID, Object itemID) throws TasteException {
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
  public List<RecommendedItem> recommend(Object userID, int howMany,
                                         Rescorer<Item> rescorer) throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }

    log.debug("Recommending items for user ID '{}'", userID);

    User theUser = getDataModel().getUser(userID);

    Set<Item> allItems = getAllOtherItems(theUser);

    TopItems.Estimator<Item> estimator = new Estimator(theUser);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, allItems, rescorer, estimator);

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

  private final class Estimator implements TopItems.Estimator<Item> {

    private final User theUser;

    private Estimator(User theUser) {
      this.theUser = theUser;
    }

    @Override
    public double estimate(Item item) throws TasteException {
      return estimatePreference(theUser, item.getID());
    }
  }

}
