/*
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

package org.apache.mahout.cf.taste.eval;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * Implementations of this interface determine the items that are considered relevant,
 * and splits data into a training and test subset, for purposes of precision/recall
 * tests as implemented by implementations of {@link RecommenderIRStatsEvaluator}.
 */
public interface RelevantItemsDataSplitter {

  /**
   * During testing, relevant items are removed from a particular users' preferences,
   * and a model is build using this user's other preferences and all other users.
   *
   * @param at                 Maximum number of items to be removed
   * @param relevanceThreshold Minimum strength of preference for an item to be considered
   *                           relevant
   * @return IDs of relevant items
   */
  FastIDSet getRelevantItemsIDs(long userID,
                                int at,
                                double relevanceThreshold,
                                DataModel dataModel) throws TasteException;

  /**
   * Adds a single user and all their preferences to the training model.
   *
   * @param userID          ID of user whose preferences we are trying to predict
   * @param relevantItemIDs IDs of items considered relevant to that user
   * @param trainingUsers   the database of training preferences to which we will
   *                        append the ones for otherUserID.
   * @param otherUserID     for whom we are adding preferences to the training model
   */
  void processOtherUser(long userID,
                        FastIDSet relevantItemIDs,
                        FastByIDMap<PreferenceArray> trainingUsers,
                        long otherUserID,
                        DataModel dataModel) throws TasteException;

}
