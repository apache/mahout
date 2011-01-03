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

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link org.apache.mahout.cf.taste.recommender.Recommender} that uses matrix factorization (a projection of users
 * and items onto a feature space)
 */
public final class SVDRecommender extends AbstractRecommender {

  private Factorization factorization;
  private final RefreshHelper refreshHelper;

  private static final Logger log = LoggerFactory.getLogger(SVDRecommender.class);

  public SVDRecommender(DataModel dataModel, Factorizer factorizer) throws TasteException {
    this(dataModel, factorizer, getDefaultCandidateItemsStrategy());
  }

  public SVDRecommender(DataModel dataModel, Factorizer factorizer, CandidateItemsStrategy candidateItemsStrategy)
      throws TasteException {
    super(dataModel, candidateItemsStrategy);
    factorization = factorizer.factorize();
    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        // TODO: train again
        return null;
      }
    });
    refreshHelper.addDependency(getDataModel());
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);

    FastIDSet possibleItemIDs = getAllOtherItems(userID);

    List<RecommendedItem> topItems = TopItems.getTopItems(howMany, possibleItemIDs.iterator(), rescorer,
        new Estimator(userID));
    log.debug("Recommendations are: {}", topItems);

    return topItems;
  }

  /**
   * a preference is estimated by computing the dot-product of the user and item feature vectors
   */
  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    double[] userFeatures = factorization.getUserFeatures(userID);
    double[] itemFeatures = factorization.getItemFeatures(itemID);
    double estimate = 0;
    for (int feature = 0; feature < userFeatures.length; feature++) {
      estimate += userFeatures[feature] * itemFeatures[feature];
    }
    return (float) estimate;
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

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
}
