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

import java.io.IOException;
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
import org.apache.mahout.cf.taste.model.PreferenceArray;
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
  private final Factorizer factorizer;
  private final PersistenceStrategy persistenceStrategy;
  private final RefreshHelper refreshHelper;

  private static final Logger log = LoggerFactory.getLogger(SVDRecommender.class);

  public SVDRecommender(DataModel dataModel, Factorizer factorizer) throws TasteException {
    this(dataModel, factorizer, getDefaultCandidateItemsStrategy(), getDefaultPersistenceStrategy());
  }

  public SVDRecommender(DataModel dataModel, Factorizer factorizer, CandidateItemsStrategy candidateItemsStrategy)
    throws TasteException {
    this(dataModel, factorizer, candidateItemsStrategy, getDefaultPersistenceStrategy());
  }

  /**
   * Create an SVDRecommender using a persistent store to cache factorizations. A factorization is loaded from the
   * store if present, otherwise a new factorization is computed and saved in the store.
   *
   * The {@link #refresh(java.util.Collection) refresh} method recomputes the factorization and overwrites the store.
   *
   * @param dataModel
   * @param factorizer
   * @param persistenceStrategy
   * @throws TasteException
   * @throws IOException
   */
  public SVDRecommender(DataModel dataModel, Factorizer factorizer, PersistenceStrategy persistenceStrategy) 
    throws TasteException {
    this(dataModel, factorizer, getDefaultCandidateItemsStrategy(), persistenceStrategy);
  }

  /**
   * Create an SVDRecommender using a persistent store to cache factorizations. A factorization is loaded from the
   * store if present, otherwise a new factorization is computed and saved in the store. 
   *
   * The {@link #refresh(java.util.Collection) refresh} method recomputes the factorization and overwrites the store.
   *
   * @param dataModel
   * @param factorizer
   * @param candidateItemsStrategy
   * @param persistenceStrategy
   *
   * @throws TasteException
   */
  public SVDRecommender(DataModel dataModel, Factorizer factorizer, CandidateItemsStrategy candidateItemsStrategy,
      PersistenceStrategy persistenceStrategy) throws TasteException {
    super(dataModel, candidateItemsStrategy);
    this.factorizer = Preconditions.checkNotNull(factorizer);
    this.persistenceStrategy = Preconditions.checkNotNull(persistenceStrategy);
    try {
      factorization = persistenceStrategy.load();
    } catch (IOException e) {
      throw new TasteException("Error loading factorization", e);
    }
    
    if (factorization == null) {
      train();
    }
    
    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        train();
        return null;
      }
    });
    refreshHelper.addDependency(getDataModel());
    refreshHelper.addDependency(factorizer);
    refreshHelper.addDependency(candidateItemsStrategy);
  }

  static PersistenceStrategy getDefaultPersistenceStrategy() {
    return new NoPersistenceStrategy();
  }

  private void train() throws TasteException {
    factorization = factorizer.factorize();
    try {
      persistenceStrategy.maybePersist(factorization);
    } catch (IOException e) {
      throw new TasteException("Error persisting factorization", e);
    }
  }
  
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");
    log.debug("Recommending items for user ID '{}'", userID);

    PreferenceArray preferencesFromUser = getDataModel().getPreferencesFromUser(userID);
    FastIDSet possibleItemIDs = getAllOtherItems(userID, preferencesFromUser);

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

  /**
   * Refresh the data model and factorization.
   */
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

}
