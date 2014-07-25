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

package org.apache.mahout.cf.taste.impl.recommender.slim;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.AllUnknownItemsCandidateItemsStrategy;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * The SLIM {@link org.apache.mahout.cf.taste.recommender.Recommender}
 * implementation that learns item-item relationships to produce
 * recommendations.
 * 
 * http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
 * 
 */
public final class SparseLinearMethodsRecommender extends AbstractRecommender {

  private SlimSolution slimSolution;
  private final Optimizer optimizer;
  private final RefreshHelper refreshHelper;

  private static final Logger log = LoggerFactory
      .getLogger(SparseLinearMethodsRecommender.class);

  /**
   * Constructor takes a data model and an optimizer that constructs the SLIM
   * solution.
   * 
   * @param dataModel
   * @param optimizer
   * @throws TasteException
   */
  public SparseLinearMethodsRecommender(DataModel dataModel, Optimizer optimizer)
      throws TasteException {
    this(dataModel, optimizer, new AllUnknownItemsCandidateItemsStrategy());
  }

  /**
   * Constructor takes a data model, an optimizer that constructs the SLIM
   * solution and a candidate items strategy object.
   * 
   * @param dataModel
   * @param optimizer
   * @param candidateItemsStrategy
   * @throws TasteException
   */
  public SparseLinearMethodsRecommender(DataModel dataModel,
      Optimizer optimizer, CandidateItemsStrategy candidateItemsStrategy)
      throws TasteException {
    super(dataModel, candidateItemsStrategy);
    this.optimizer = Preconditions.checkNotNull(optimizer);
    train();

    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        train();
        return null;
      }
    });
    refreshHelper.addDependency(getDataModel());
    refreshHelper.addDependency(optimizer);
  }

  private void train() throws TasteException {
    slimSolution = optimizer.findSolution();
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany,
      IDRescorer rescorer, boolean includeKnownItems) throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);

    PreferenceArray preferencesFromUser = getDataModel()
        .getPreferencesFromUser(userID);
    FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser,
        includeKnownItems);

    Estimator estimator = new Estimator(userID);
    List<RecommendedItem> topItems = TopItems.getTopItems(howMany,
        possibleItemIDs.iterator(), rescorer, estimator);
    log.debug("Recommendations are: {}", topItems);

    return topItems;
  }

  public Optimizer getOptimizer() {
    return optimizer;
  }

  public SlimSolution getSolution() {
    return slimSolution;
  }

  /**
   * Estimate userID's preference for itemID using userID's current item
   * preferences and their relationship with itemID.
   */
  @Override
  public float estimatePreference(long userID, long itemID)
      throws TasteException {
    DataModel model = getDataModel();

    PreferenceArray userPrefs = model.getPreferencesFromUser(userID);
    int size = userPrefs.length();

    int itemIndex = slimSolution.itemIndex(itemID);
    Matrix itemFeatures = slimSolution.getItemWeights();
    float estimate = 0;
    for (int i = 0; i < size; i++) {
      long item2ID = userPrefs.getItemID(i);
      float userPref = 1.0f;
      if (model.hasPreferenceValues()) {
        userPref = userPrefs.getValue(i);
      }
      int item2Index = slimSolution.itemIndex(item2ID);
      estimate += userPref * itemFeatures.getQuick(itemIndex, item2Index);
    }

    return estimate;
  }

  private final class Estimator implements TopItems.Estimator<Long> {

    private final long theUserID;

    private Estimator(long theUserID) {
      this.theUserID = theUserID;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      float estimate = estimatePreference(theUserID, itemID);
      return estimate;
    }

  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

}
