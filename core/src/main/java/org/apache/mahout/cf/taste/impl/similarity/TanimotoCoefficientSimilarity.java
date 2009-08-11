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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;

/**
 * <p>An implementation of a "similarity" based on the
 * <a href="http://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_coefficient_.28extended_Jaccard_coefficient.29">
 * Tanimoto coefficient</a>, or extended <a href="http://en.wikipedia.org/wiki/Jaccard_index">Jaccard
 * coefficient</a>.</p>
 *
 * <p>This is intended for "binary" data sets where a user either expresses a generic "yes" preference for an item or
 * has no preference. The actual preference values do not matter here, only their presence or absence.</p>
 *
 * <p>The value returned is in [0,1].</p>
 */
public final class TanimotoCoefficientSimilarity implements UserSimilarity, ItemSimilarity {

  private final DataModel dataModel;

  public TanimotoCoefficientSimilarity(DataModel dataModel) {
    this.dataModel = dataModel;
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

    FastIDSet xPrefs = dataModel.getItemIDsFromUser(userID1);
    FastIDSet yPrefs = dataModel.getItemIDsFromUser(userID2);

    if (xPrefs.isEmpty() && yPrefs.isEmpty()) {
      return Double.NaN;
    }
    if (xPrefs.isEmpty() || yPrefs.isEmpty()) {
      return 0.0;
    }

    int intersectionSize = xPrefs.intersectionSize(yPrefs);
    if (intersectionSize == 0) {
      return Double.NaN;
    }

    int unionSize = xPrefs.size() + yPrefs.size() - intersectionSize;

    return (double) intersectionSize / (double) unionSize;
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    int preferring1and2 = dataModel.getNumUsersWithPreferenceFor(itemID1, itemID2);
    int preferring1 = dataModel.getNumUsersWithPreferenceFor(itemID1);
    int preferring2 = dataModel.getNumUsersWithPreferenceFor(itemID2);
    return (double) preferring1and2 / (double) (preferring1 + preferring2 - preferring1and2);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, dataModel);
  }

  @Override
  public String toString() {
    return "TanimotoCoefficientSimilarity[dataModel:" + dataModel + ']';
  }

}