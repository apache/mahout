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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.similarity.AbstractItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;

final class DummySimilarity extends AbstractItemSimilarity implements UserSimilarity {

  DummySimilarity(DataModel dataModel) {
    super(dataModel);
  }
  
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    DataModel dataModel = getDataModel();
    return 1.0 / (1.0 + Math.abs(dataModel.getPreferencesFromUser(userID1).get(0).getValue()
                                 - dataModel.getPreferencesFromUser(userID2).get(0).getValue()));
  }
  
  @Override
  public double itemSimilarity(long itemID1, long itemID2) {
    // Make up something wacky
    return 1.0 / (1.0 + Math.abs(itemID1 - itemID2));
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) {
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = itemSimilarity(itemID1, itemID2s[i]);
    }
    return result;
  }
  
  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
  // do nothing
  }
  
}
