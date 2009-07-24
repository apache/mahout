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
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.BooleanPrefUser;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;

/**
 * <p>Variant of {@link TanimotoCoefficientSimilarity} which is appropriate for use with the "boolean" classes like
 * {@link BooleanPrefUser}.</p>
 *
 * <p>If you need an {@link org.apache.mahout.cf.taste.similarity.ItemSimilarity}, just use {@link
 * org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity}, even with "boolean" classes.</p>
 */
public final class BooleanTanimotoCoefficientSimilarity implements UserSimilarity {

  private final DataModel dataModel;

  public BooleanTanimotoCoefficientSimilarity(DataModel dataModel) {
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
  public double userSimilarity(User user1, User user2) {
    FastSet<Comparable<?>> prefs1 = ((BooleanPrefUser) user1).getItemIDs();
    FastSet<Comparable<?>> prefs2 = ((BooleanPrefUser) user2).getItemIDs();
    int prefs1Size = prefs1.size();
    int prefs2Size = prefs2.size();
    int intersectionSize = prefs1Size < prefs2Size ?
        prefs2.intersectionSize(prefs1) :
        prefs1.intersectionSize(prefs2);
    int unionSize = prefs1Size + prefs2Size - intersectionSize;
    return (double) intersectionSize / (double) unionSize;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(alreadyRefreshed, dataModel);
  }

  @Override
  public String toString() {
    return "BooleanTanimotoCoefficientSimilarity[dataModel:" + dataModel + ']';
  }

}