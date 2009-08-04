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

package org.apache.mahout.cf.taste.impl.recommender.knn;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * <p>The weights to compute the final predicted preferences are calculated using linear interpolation, through an
 * {@link Optimizer}. This algorithm is based in the paper of Robert M. Bell and Yehuda Koren in ICDM '07.</p>
 */
public final class KnnItemBasedRecommender extends GenericItemBasedRecommender {

  private final Optimizer optimizer;
  private final int neighborhoodSize;

  public KnnItemBasedRecommender(DataModel dataModel,
                                 ItemSimilarity similarity,
                                 Optimizer optimizer,
                                 int neighborhoodSize) {
    super(dataModel, similarity);
    this.optimizer = optimizer;
    this.neighborhoodSize = neighborhoodSize;
  }

  private List<RecommendedItem> mostSimilarItems(Comparable<?> itemID,
                                                 Iterable<Comparable<?>> allItemIDs,
                                                 int howMany,
                                                 Rescorer<Pair<Comparable<?>, Comparable<?>>> rescorer)
          throws TasteException {
    TopItems.Estimator<Comparable<?>> estimator = new MostSimilarEstimator(itemID, getSimilarity(), rescorer);
    return TopItems.getTopItems(howMany, allItemIDs, null, estimator);
  }


  private double[] getInterpolations(Comparable<?> itemID, Comparable<?> userID, List<Comparable<?>> itemNeighborhood)
          throws TasteException {

    int k = itemNeighborhood.size();
    double[][] A = new double[k][k];
    double[] b = new double[k];
    int i = 0;

    DataModel dataModel = getDataModel();

    int numUsers = getDataModel().getNumUsers();
    for (Comparable<?> iitem : itemNeighborhood) {
      PreferenceArray iPrefs = getDataModel().getPreferencesForItem(iitem);
      int iSize = iPrefs.length();
      int j = 0;
      for (Comparable<?> jitem : itemNeighborhood) {
        double value = 0.0;
        for (int pi = 0; pi < iSize; pi++) {
          Comparable<?> v = iPrefs.getUserID(pi);
          if (v.equals(userID)) {
            continue;
          }
          Float pj = dataModel.getPreferenceValue(userID, jitem);
          if (pj != null) {
            value += iPrefs.getValue(pi) * pj;
          }
        }
        A[i][j] = value / numUsers;
        j++;
      }
      i++;
    }

    PreferenceArray iPrefs = getDataModel().getPreferencesForItem(itemID);
    int iSize = iPrefs.length();
    i = 0;
    for (Comparable<?> jitem : itemNeighborhood) {
      double value = 0.0;
      for (int pi = 0; pi < iSize; pi++) {
        Comparable<?> v = iPrefs.getUserID(pi);
        if (v.equals(userID)) {
          continue;
        }
        Float pj = dataModel.getPreferenceValue(userID, jitem);
        if (pj != null) {
          value += iPrefs.getValue(pi) * pj;
        }
      }
      b[i] = value / numUsers;
      i++;
    }

    return optimizer.optimize(A, b);
  }

  @Override
  protected float doEstimatePreference(Comparable<?> theUserID, Comparable<?> itemID) throws TasteException {

    DataModel dataModel = getDataModel();
    Collection<Comparable<?>> allItemIDs = new FastSet<Comparable<?>>();
    PreferenceArray prefs = dataModel.getPreferencesFromUser(theUserID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      allItemIDs.add(prefs.getItemID(i));
    }
    allItemIDs.remove(itemID);

    List<RecommendedItem> mostSimilar = mostSimilarItems(itemID, allItemIDs, neighborhoodSize, null);
    List<Comparable<?>> theNeighborhood = new ArrayList<Comparable<?>>(mostSimilar.size());
    for (RecommendedItem rec : mostSimilar) {
      theNeighborhood.add(rec.getItemID());
    }


    double[] weights = getInterpolations(itemID, theUserID, theNeighborhood);

    int i = 0;
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (Comparable<?> jitem : theNeighborhood) {

      Float pref = dataModel.getPreferenceValue(theUserID, jitem);

      if (pref != null) {
        preference += pref * weights[i];
        totalSimilarity += weights[i];
      }
      i++;

    }
    return totalSimilarity == 0.0 ? Float.NaN : (float) (preference / totalSimilarity);
  }

}
