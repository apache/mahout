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

package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * <p>
 * An implementation of a "similarity" based on the <a
 * href="http://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_coefficient_.28extended_Jaccard_coefficient.29">
 * Tanimoto coefficient</a>, or extended <a href="http://en.wikipedia.org/wiki/Jaccard_index">Jaccard
 * coefficient</a>.
 * </p>
 * 
 * <p>
 * This is intended for "binary" data sets where a user either expresses a generic "yes" preference for an
 * item or has no preference. The actual preference values do not matter here, only their presence or absence.
 * </p>
 * 
 * <p>
 * The value returned is in [0,1].
 * </p>
 */
public final class TanimotoCoefficientSimilarity extends AbstractItemSimilarity implements UserSimilarity {

  public TanimotoCoefficientSimilarity(DataModel dataModel) {
    super(dataModel);
  }
  
  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {

    DataModel dataModel = getDataModel();
    FastIDSet xPrefs = dataModel.getItemIDsFromUser(userID1);
    FastIDSet yPrefs = dataModel.getItemIDsFromUser(userID2);

    int xPrefsSize = xPrefs.size();
    int yPrefsSize = yPrefs.size();
    if (xPrefsSize == 0 && yPrefsSize == 0) {
      return Double.NaN;
    }
    if (xPrefsSize == 0 || yPrefsSize == 0) {
      return 0.0;
    }
    
    int intersectionSize =
        xPrefsSize < yPrefsSize ? yPrefs.intersectionSize(xPrefs) : xPrefs.intersectionSize(yPrefs);
    if (intersectionSize == 0) {
      return Double.NaN;
    }
    
    int unionSize = xPrefsSize + yPrefsSize - intersectionSize;
    
    return (double) intersectionSize / (double) unionSize;
  }
  
  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    int preferring1 = getDataModel().getNumUsersWithPreferenceFor(itemID1);
    return doItemSimilarity(itemID1, itemID2, preferring1);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    int preferring1 = getDataModel().getNumUsersWithPreferenceFor(itemID1);
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = doItemSimilarity(itemID1, itemID2s[i], preferring1);
    }
    return result;
  }

  private double doItemSimilarity(long itemID1, long itemID2, int preferring1) throws TasteException {
    DataModel dataModel = getDataModel();
    int preferring1and2 = dataModel.getNumUsersWithPreferenceFor(itemID1, itemID2);
    if (preferring1and2 == 0) {
      return Double.NaN;
    }
    int preferring2 = dataModel.getNumUsersWithPreferenceFor(itemID2);
    return (double) preferring1and2 / (double) (preferring1 + preferring2 - preferring1and2);
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, getDataModel());
  }
  
  @Override
  public String toString() {
    return "TanimotoCoefficientSimilarity[dataModel:" + getDataModel() + ']';
  }
  
}
