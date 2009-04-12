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
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import java.util.ArrayList;
import java.util.List;
import java.util.Collection;

/**
 * <p>The weights to compute the final predicted preferences are calculated
 * using linear interpolation, through an {@link Optimizer}. This algorithm is based in
 * the paper of Robert M. Bell and Yehuda Koren in ICDM '07.</p>
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

  private List<RecommendedItem> mostSimilarItems(Object itemID,
                                                 Iterable<Item> allItems,
                                                 int howMany,
                                                 Rescorer<Pair<Item, Item>> rescorer) throws TasteException {
    Item toItem = getDataModel().getItem(itemID);
    TopItems.Estimator<Item> estimator = new MostSimilarEstimator(toItem, getSimilarity(), rescorer);
    return TopItems.getTopItems(howMany, allItems, null, estimator);
  }


  private double[] getInterpolations(Item item, User theUser, List<Item> itemNeighborhood) throws TasteException {

    int k = itemNeighborhood.size();
    double[][] A = new double[k][k];
    double[] b = new double[k];
    int i = 0;

    int numUsers = getDataModel().getNumUsers();
    for (Item iitem : itemNeighborhood) {
      Preference[] iPrefs = getDataModel().getPreferencesForItemAsArray(iitem.getID());
      int j = 0;
      for (Item jitem : itemNeighborhood) {
        double value = 0.0;
        for (Preference pi : iPrefs) {
          User v = pi.getUser();
          if (v.equals(theUser)) {
            continue;
          }
          Preference pj = v.getPreferenceFor(jitem.getID());
          if (pj != null) {
            value += pi.getValue() * pj.getValue();
          }
        }
        A[i][j] = value / numUsers;
        j++;
      }
      i++;
    }

    Preference[] iPrefs = getDataModel().getPreferencesForItemAsArray(item.getID());
    i = 0;
    for (Item jitem : itemNeighborhood) {
      double value = 0.0;
      for (Preference pi : iPrefs) {
        User v = pi.getUser();
        if (v.equals(theUser)) {
          continue;
        }
        Preference pj = v.getPreferenceFor(jitem.getID());
        if (pj != null) {
          value += pi.getValue() * pj.getValue();
        }
      }
      b[i] = value / numUsers;
      i++;
    }

    return optimizer.optimize(A, b);
  }

  @Override
  protected double doEstimatePreference(User theUser, Item item) throws TasteException {

    Collection<Item> allItems = new FastSet<Item>();
    for(Preference pref: theUser.getPreferencesAsArray()){
      allItems.add(pref.getItem());
    }
    allItems.remove(item);

    List<RecommendedItem> mostSimilar = mostSimilarItems(item.getID(), allItems, neighborhoodSize, null);
    List<Item> theNeighborhood = new ArrayList<Item>(mostSimilar.size());
    for (RecommendedItem rec: mostSimilar) {
      theNeighborhood.add(rec.getItem());
    }


    double[] weights = getInterpolations(item, theUser, theNeighborhood);

    int i = 0;
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (Item jitem : theNeighborhood) {

      Preference pref = theUser.getPreferenceFor(jitem.getID());

      if (pref != null) {
        preference += pref.getValue() * weights[i];
        totalSimilarity += weights[i];
      }
      i++;

    }
    return totalSimilarity == 0.0 ? Double.NaN : preference / totalSimilarity;
  }

}
