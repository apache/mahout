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

import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/**
 * returns all items that have not been rated by the user and that were preferred by another user
 * that has preferred at least one item that the current user has preferred too
 */
public final class PreferredItemsNeighborhoodCandidateItemsStrategy implements CandidateItemsStrategy {

  @Override
  public FastIDSet getCandidateItems(long userID, DataModel dataModel) throws TasteException {
    FastIDSet possibleItemsIDs = new FastIDSet();
    FastIDSet itemIDs = dataModel.getItemIDsFromUser(userID);
    LongPrimitiveIterator itemIDIterator = itemIDs.iterator();
    while (itemIDIterator.hasNext()) {
      long itemID = itemIDIterator.nextLong();
      PreferenceArray prefs2 = dataModel.getPreferencesForItem(itemID);
      int size2 = prefs2.length();
      for (int j = 0; j < size2; j++) {
        possibleItemsIDs.addAll(dataModel.getItemIDsFromUser(prefs2.getUserID(j)));
      }
    }
    possibleItemsIDs.removeAll(itemIDs);
    return possibleItemsIDs;
  }
  
}
