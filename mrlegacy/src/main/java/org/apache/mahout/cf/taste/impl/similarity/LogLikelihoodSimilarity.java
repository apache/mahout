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
import org.apache.mahout.math.stats.LogLikelihood;

/**
 * See <a href="http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.5962">
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.14.5962</a> and
 * <a href="http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html">
 * http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html</a>.
 */
public final class LogLikelihoodSimilarity extends AbstractItemSimilarity implements UserSimilarity {

  public LogLikelihoodSimilarity(DataModel dataModel) {
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
    FastIDSet prefs1 = dataModel.getItemIDsFromUser(userID1);
    FastIDSet prefs2 = dataModel.getItemIDsFromUser(userID2);
    
    long prefs1Size = prefs1.size();
    long prefs2Size = prefs2.size();
    long intersectionSize =
        prefs1Size < prefs2Size ? prefs2.intersectionSize(prefs1) : prefs1.intersectionSize(prefs2);
    if (intersectionSize == 0) {
      return Double.NaN;
    }
    long numItems = dataModel.getNumItems();
    double logLikelihood =
        LogLikelihood.logLikelihoodRatio(intersectionSize,
                                         prefs2Size - intersectionSize,
                                         prefs1Size - intersectionSize,
                                         numItems - prefs1Size - prefs2Size + intersectionSize);
    return 1.0 - 1.0 / (1.0 + logLikelihood);
  }
  
  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    DataModel dataModel = getDataModel();
    long preferring1 = dataModel.getNumUsersWithPreferenceFor(itemID1);
    long numUsers = dataModel.getNumUsers();
    return doItemSimilarity(itemID1, itemID2, preferring1, numUsers);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    DataModel dataModel = getDataModel();
    long preferring1 = dataModel.getNumUsersWithPreferenceFor(itemID1);
    long numUsers = dataModel.getNumUsers();
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = doItemSimilarity(itemID1, itemID2s[i], preferring1, numUsers);
    }
    return result;
  }

  private double doItemSimilarity(long itemID1, long itemID2, long preferring1, long numUsers) throws TasteException {
    DataModel dataModel = getDataModel();
    long preferring1and2 = dataModel.getNumUsersWithPreferenceFor(itemID1, itemID2);
    if (preferring1and2 == 0) {
      return Double.NaN;
    }
    long preferring2 = dataModel.getNumUsersWithPreferenceFor(itemID2);
    double logLikelihood =
        LogLikelihood.logLikelihoodRatio(preferring1and2,
                                         preferring2 - preferring1and2,
                                         preferring1 - preferring1and2,
                                         numUsers - preferring1 - preferring2 + preferring1and2);
    return 1.0 - 1.0 / (1.0 + logLikelihood);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, getDataModel());
  }
  
  @Override
  public String toString() {
    return "LogLikelihoodSimilarity[dataModel:" + getDataModel() + ']';
  }
  
}
