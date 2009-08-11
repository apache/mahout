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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public abstract class AbstractRecommender implements Recommender {

  private static final Logger log = LoggerFactory.getLogger(AbstractRecommender.class);

  private final DataModel dataModel;

  protected AbstractRecommender(DataModel dataModel) {
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    this.dataModel = dataModel;
  }

  /**
   * <p>Default implementation which just calls {@link Recommender#recommend(long, int,
   * org.apache.mahout.cf.taste.recommender.Rescorer)}, with a {@link org.apache.mahout.cf.taste.recommender.Rescorer}
   * that does nothing.</p>
   */
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
    return recommend(userID, howMany, null);
  }

  /**
   * <p>Default implementation which just calls {@link DataModel#setPreference(long, long, float)}.</p>
   *
   * @throws IllegalArgumentException if userID or itemID is <code>null</code>, or if value is {@link Double#NaN}
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    if (Double.isNaN(value)) {
      throw new IllegalArgumentException("Invalid value: " + value);
    }
    log.debug("Setting preference for user {}, item {}", userID, itemID);    
    dataModel.setPreference(userID, itemID, value);
  }

  /**
   * <p>Default implementation which just calls {@link DataModel#removePreference(long, long)} (Object,
   * Object)}.</p>
   *
   * @throws IllegalArgumentException if userID or itemID is <code>null</code>
   */
  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    log.debug("Remove preference for user '{}', item '{}'", userID, itemID);
    dataModel.removePreference(userID, itemID);
  }

  @Override
  public DataModel getDataModel() {
    return dataModel;
  }

  /**
   * @param theUserID ID of user being evaluated
   * @return all items in the {@link DataModel} for which the user has not expressed a preference
   * @throws TasteException if an error occurs while listing items
   */
  protected FastIDSet getAllOtherItems(long theUserID) throws TasteException {
    FastIDSet allItemIDs = new FastIDSet(dataModel.getNumItems());
    LongPrimitiveIterator it = dataModel.getItemIDs();
    while (it.hasNext()) {
      allItemIDs.add(it.nextLong());
    }
    PreferenceArray prefs = dataModel.getPreferencesFromUser(theUserID);
    int size = prefs.length();
    for (int i = 0; i < size; i++) {
      allItemIDs.remove(prefs.getItemID(i));
    }
    return allItemIDs;
  }

}
