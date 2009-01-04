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

package org.apache.mahout.cf.taste.example.bookcrossing;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.CachingRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.util.Collection;
import java.util.List;

/**
 * A simple {@link Recommender} implemented for the Book Crossing demo.
 * See the <a href="http://www.informatik.uni-freiburg.de/~cziegler/BX/">Book Crossing site</a>.
 */
public final class BookCrossingRecommender implements Recommender {

  private final Recommender recommender;

  public BookCrossingRecommender(DataModel dataModel, BookCrossingDataModel bcModel) throws TasteException {
    UserSimilarity similarity = new GeoUserSimilarity(bcModel);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(5, similarity, dataModel);
    recommender = new CachingRecommender(new GenericUserBasedRecommender(dataModel, neighborhood, similarity));
  }

  @Override
  public List<RecommendedItem> recommend(Object userID, int howMany) throws TasteException {
    return recommender.recommend(userID, howMany);
  }

  @Override
  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
          throws TasteException {
    return recommender.recommend(userID, howMany, rescorer);
  }

  @Override
  public double estimatePreference(Object userID, Object itemID) throws TasteException {
    return recommender.estimatePreference(userID, itemID);
  }

  @Override
  public void setPreference(Object userID, Object itemID, double value) throws TasteException {
    recommender.setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(Object userID, Object itemID) throws TasteException {
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
    return "BookCrossingRecommender[recommender:" + recommender + ']';
  }

}