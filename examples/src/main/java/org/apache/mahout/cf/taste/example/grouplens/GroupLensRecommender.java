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

package org.apache.mahout.cf.taste.example.grouplens;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

/**
 * A simple {@link Recommender} implemented for the GroupLens demo.
 */
public final class GroupLensRecommender implements Recommender {

  private final Recommender recommender;

  /**
   * @throws IOException if an error occurs while creating the {@link GroupLensDataModel}
   * @throws TasteException if an error occurs while initializing this {@link GroupLensRecommender}
   */
  public GroupLensRecommender() throws IOException, TasteException {
    this(new GroupLensDataModel());
  }

  /**
   * <p>Alternate constructor that takes a {@link DataModel} argument, which allows this {@link Recommender}
   * to be used with the {@link org.apache.mahout.cf.taste.eval.RecommenderEvaluator} framework.</p>
   *
   * @param dataModel data model
   * @throws TasteException if an error occurs while initializing this {@link GroupLensRecommender}
   */
  public GroupLensRecommender(DataModel dataModel) throws TasteException {
    recommender = new CachingRecommender(new SlopeOneRecommender(dataModel));
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
    return recommender.recommend(userID, howMany);
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer)
          throws TasteException {
    return recommender.recommend(userID, howMany, rescorer);
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    return recommender.estimatePreference(userID, itemID);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    recommender.setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    recommender.removePreference(userID, itemID);
  }

  @Override
  public DataModel getDataModel() {
    return recommender.getDataModel();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    recommender.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "GroupLensRecommender[recommender:" + recommender + ']';
  }

}
