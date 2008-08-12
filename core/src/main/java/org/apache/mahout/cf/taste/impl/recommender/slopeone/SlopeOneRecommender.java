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

package org.apache.mahout.cf.taste.impl.recommender.slopeone;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.slopeone.DiffStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Collection;

/**
 * <p>A basic "slope one" recommender. (See an <a href="http://www.daniel-lemire.com/fr/abstracts/SDM2005.html">
 * excellent summary here</a> for example.) This {@link org.apache.mahout.cf.taste.recommender.Recommender} is especially
 * suitable when user preferences are updating frequently as it can incorporate this information without
 * expensive recomputation.</p>
 *
 * <p>This implementation can also be used as a "weighted slope one" recommender.</p>
 */
public final class SlopeOneRecommender extends AbstractRecommender {

  private static final Logger log = LoggerFactory.getLogger(SlopeOneRecommender.class);

  private final boolean weighted;
  private final boolean stdDevWeighted;
  private final DiffStorage diffStorage;

  /**
   * <p>Creates a default (weighted) {@link SlopeOneRecommender} based on the given {@link DataModel}.</p>
   *
   * @param dataModel data model
   */
  public SlopeOneRecommender(DataModel dataModel) throws TasteException {
    this(dataModel,
         Weighting.WEIGHTED,
         Weighting.WEIGHTED,
         new MemoryDiffStorage(dataModel, Weighting.WEIGHTED, false, Long.MAX_VALUE));
  }

  /**
   * <p>Creates a {@link SlopeOneRecommender} based on the given {@link DataModel}.</p>
   *
   * <p>If <code>weighted</code> is set, acts as a weighted slope one recommender.
   * This implementation also includes an experimental "standard deviation" weighting which weights
   * item-item ratings diffs with lower standard deviation more highly, on the theory that they are more
   * reliable.</p>
   *
   * @param weighting if {@link Weighting#WEIGHTED}, acts as a weighted slope one recommender
   * @param stdDevWeighting use optional standard deviation weighting of diffs
   * @throws IllegalArgumentException if <code>diffStorage</code> is null, or stdDevWeighted is set
   * when weighted is not set
   */
  public SlopeOneRecommender(DataModel dataModel,
                             Weighting weighting,
                             Weighting stdDevWeighting,
                             DiffStorage diffStorage) {
    super(dataModel);
    if (stdDevWeighting == Weighting.WEIGHTED && weighting == Weighting.UNWEIGHTED) {
      throw new IllegalArgumentException("weighted required when stdDevWeighted is set");
    }
    if (diffStorage == null) {
      throw new IllegalArgumentException("diffStorage is null");
    }
    this.weighted = weighting == Weighting.WEIGHTED;
    this.stdDevWeighted = stdDevWeighting == Weighting.WEIGHTED;
    this.diffStorage = diffStorage;
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

    User theUser = getDataModel().getUser(userID);
    Set<Item> allItems = diffStorage.getRecommendableItems(userID);

    TopItems.Estimator<Item> estimator = new Estimator(theUser);

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
    return doEstimatePreference(theUser, itemID);
  }

  private double doEstimatePreference(User theUser, Object itemID) throws TasteException {
    double count = 0.0;
    double totalPreference = 0.0;
    Preference[] prefs = theUser.getPreferencesAsArray();
    RunningAverage[] averages = diffStorage.getDiffs(theUser.getID(), itemID, prefs);
    for (int i = 0; i < prefs.length; i++) {
      RunningAverage averageDiff = averages[i];
      if (averageDiff != null) {
        Preference pref = prefs[i];
        double averageDiffValue = averageDiff.getAverage();
        if (weighted) {
          double weight = (double) averageDiff.getCount();
          if (stdDevWeighted) {
            double stdev = ((RunningAverageAndStdDev) averageDiff).getStandardDeviation();
            if (!Double.isNaN(stdev)) {
              weight /= 1.0 + stdev;
            }
            // If stdev is NaN, then it is because count is 1. Because we're weighting by count,
            // the weight is already relatively low. We effectively assume stdev is 0.0 here and
            // that is reasonable enough. Otherwise, dividing by NaN would yield a weight of NaN
            // and disqualify this pref entirely
            // (Thanks Daemmon)
          }
          totalPreference += weight * (pref.getValue() + averageDiffValue);
          count += weight;
        } else {
          totalPreference += pref.getValue() + averageDiffValue;
          count += 1.0;
        }
      }
    }
    if (count <= 0.0) {
      RunningAverage itemAverage = diffStorage.getAverageItemPref(itemID);
      return itemAverage == null ? Double.NaN : itemAverage.getAverage();
    } else {
      return totalPreference / count;
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
    diffStorage.updateItemPref(itemID, prefDelta, false);
  }

  @Override
  public void removePreference(Object userID, Object itemID) throws TasteException {
    DataModel dataModel = getDataModel();
    User theUser = dataModel.getUser(userID);
    Preference oldPref = theUser.getPreferenceFor(itemID);
    super.removePreference(userID, itemID);
    if (oldPref != null) {
      diffStorage.updateItemPref(itemID, oldPref.getValue(), true);
    }
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, diffStorage);
  }

  @Override
  public String toString() {
    return "SlopeOneRecommender[weighted:" + weighted + ", stdDevWeighted:" + stdDevWeighted +
           ", diffStorage:" + diffStorage + ']';
  }

  private final class Estimator implements TopItems.Estimator<Item> {

    private final User theUser;

    private Estimator(User theUser) {
      this.theUser = theUser;
    }

    public double estimate(Item item) throws TasteException {
      return doEstimatePreference(theUser, item.getID());
    }
  }

}
