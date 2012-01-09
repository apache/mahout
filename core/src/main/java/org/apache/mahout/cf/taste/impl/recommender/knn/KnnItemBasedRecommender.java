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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.MostSimilarItemsCandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.LongPair;

/**
 * <p>
 * The weights to compute the final predicted preferences are calculated using linear interpolation, through
 * an {@link Optimizer}. This algorithm is based in the paper of Robert M. Bell and Yehuda Koren in ICDM '07.
 * </p>
 */
public final class KnnItemBasedRecommender extends GenericItemBasedRecommender {
  
  private static final double BETA = 500.0;

  private final Optimizer optimizer;
  private final int neighborhoodSize;
  
  public KnnItemBasedRecommender(DataModel dataModel,
                                 ItemSimilarity similarity,
                                 Optimizer optimizer,
                                 CandidateItemsStrategy candidateItemsStrategy,
                                 MostSimilarItemsCandidateItemsStrategy mostSimilarItemsCandidateItemsStrategy,
                                 int neighborhoodSize) {
    super(dataModel, similarity, candidateItemsStrategy, mostSimilarItemsCandidateItemsStrategy);
    this.optimizer = optimizer;
    this.neighborhoodSize = neighborhoodSize;
  }

  public KnnItemBasedRecommender(DataModel dataModel,
                                 ItemSimilarity similarity,
                                 Optimizer optimizer,
                                 int neighborhoodSize) {
    this(dataModel, similarity, optimizer, getDefaultCandidateItemsStrategy(),
        getDefaultMostSimilarItemsCandidateItemsStrategy(), neighborhoodSize);
  }
  
  private List<RecommendedItem> mostSimilarItems(long itemID,
                                                 LongPrimitiveIterator possibleItemIDs,
                                                 int howMany,
                                                 Rescorer<LongPair> rescorer) throws TasteException {
    TopItems.Estimator<Long> estimator = new MostSimilarEstimator(itemID, getSimilarity(), rescorer);
    return TopItems.getTopItems(howMany, possibleItemIDs, null, estimator);
  }
  
  private double[] getInterpolations(long itemID, 
                                     long[] itemNeighborhood,
                                     Collection<Long> usersRatedNeighborhood) throws TasteException {
    
    int length = 0;
    for (int i = 0; i < itemNeighborhood.length; i++) {
      if (itemNeighborhood[i] == itemID) {
        itemNeighborhood[i] = -1;
        length = itemNeighborhood.length - 1;
        break;
      }
    }
    
    int k = length;
    double[][] aMatrix = new double[k][k];
    double[] b = new double[k];
    int i = 0;
    
    DataModel dataModel = getDataModel();
    
    int numUsers = usersRatedNeighborhood.size();
    for (long iitem : itemNeighborhood) {
      if (iitem == -1) {
        break;
      }
      int j = 0;
      double value = 0.0;
      for (long jitem : itemNeighborhood) {
      if (jitem == -1) {
        continue;
      }
      for (long user : usersRatedNeighborhood) {
        float prefVJ = dataModel.getPreferenceValue(user, iitem);
        float prefVK = dataModel.getPreferenceValue(user, jitem);
          value += prefVJ * prefVK;
        }
        aMatrix[i][j] = value/numUsers;
        j++;
      }
      i++;
    }
    
    i = 0;
    for (long jitem : itemNeighborhood) {
      if (jitem == -1) {
        break;
      }
      double value = 0.0;
      for (long user : usersRatedNeighborhood) {
        float prefVJ = dataModel.getPreferenceValue(user, jitem);
        float prefVI = dataModel.getPreferenceValue(user, itemID);
        value += prefVJ * prefVI;
      }
      b[i] = value / numUsers;
      i++;
    }
    
    // Find the larger diagonal and calculate the average
    double avgDiagonal = 0.0;
    if (k > 1) {
      double diagonalA = 0.0;
      for (i = 0; i < k; i++) {
        diagonalA += aMatrix[i][i];
      }
      double diagonalB = 0.0;
      for (i = k - 1; i >= 0; i--) {
        for (int j = 0; j < k; j++) {
          diagonalB += aMatrix[i--][j];
        }
      }
      avgDiagonal = Math.max(diagonalA, diagonalB) / k;
    }
    // Calculate the average of non-diagonal values
    double avgMatrixA = 0.0;
    double avgVectorB = 0.0;
    for (i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        if (i != j || k <= 1) {
          avgMatrixA += aMatrix[i][j];
        }
      }
      avgVectorB += b[i];
    }
    if (k > 1) {
      avgMatrixA /= k * k - k;
    }
    avgVectorB /= k;

    double numUsersPlusBeta = numUsers + BETA;
    for (i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        double average;
        if (i == j && k > 1) {
          average = avgDiagonal;
        } else {
          average = avgMatrixA;
        }
        aMatrix[i][j] = (numUsers * aMatrix[i][j] + BETA * average) / numUsersPlusBeta;
      }
      b[i] = (numUsers * b[i] + BETA * avgVectorB) / numUsersPlusBeta;
    }

    return optimizer.optimize(aMatrix, b);
  }
  
  @Override
  protected float doEstimatePreference(long theUserID, PreferenceArray preferencesFromUser, long itemID)
    throws TasteException {
    
    DataModel dataModel = getDataModel();
    int size = preferencesFromUser.length();
    FastIDSet possibleItemIDs = new FastIDSet(size);
    for (int i = 0; i < size; i++) {
      possibleItemIDs.add(preferencesFromUser.getItemID(i));
    }
    possibleItemIDs.remove(itemID);
    
    List<RecommendedItem> mostSimilar = mostSimilarItems(itemID, possibleItemIDs.iterator(),
      neighborhoodSize, null);
    long[] theNeighborhood = new long[mostSimilar.size() + 1];
    theNeighborhood[0] = -1;
  
    List<Long> usersRatedNeighborhood = new ArrayList<Long>();
    int nOffset = 0;
    for (RecommendedItem rec : mostSimilar) {
      theNeighborhood[nOffset++] = rec.getItemID();
    }
    
    if (!mostSimilar.isEmpty()) {
      theNeighborhood[mostSimilar.size()] = itemID;
      for (int i = 0; i < theNeighborhood.length; i++) {
        PreferenceArray usersNeighborhood = dataModel.getPreferencesForItem(theNeighborhood[i]);
        int size1 = usersRatedNeighborhood.isEmpty() ? usersNeighborhood.length() : usersRatedNeighborhood.size();
        for (int j = 0; j < size1; j++) {
          if (i == 0) {
            usersRatedNeighborhood.add(usersNeighborhood.getUserID(j));
          } else {
            if (j >= usersRatedNeighborhood.size()) {
              break;
            }
            long index = usersRatedNeighborhood.get(j);
            if (!usersNeighborhood.hasPrefWithUserID(index) || index == theUserID) {
              usersRatedNeighborhood.remove(index);
              j--;
            }
          }
        }
      }
    }

    double[] weights = null;
    if (!mostSimilar.isEmpty()) {
      weights = getInterpolations(itemID, theNeighborhood, usersRatedNeighborhood);
    }
    
    int i = 0;
    double preference = 0.0;
    double totalSimilarity = 0.0;
    for (long jitem : theNeighborhood) {
      
      Float pref = dataModel.getPreferenceValue(theUserID, jitem);
      
      if (pref != null) {
        double weight = weights[i];
        preference += pref * weight;
        totalSimilarity += weight;
      }
      i++;
      
    }
    return totalSimilarity == 0.0 ? Float.NaN : (float) (preference / totalSimilarity);
  }
  
}
