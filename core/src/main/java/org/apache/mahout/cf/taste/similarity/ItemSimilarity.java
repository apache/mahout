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

package org.apache.mahout.cf.taste.similarity;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>
 * Implementations of this interface define a notion of similarity between two items. Implementations should
 * return values in the range -1.0 to 1.0, with 1.0 representing perfect similarity.
 * </p>
 * 
 * @see UserSimilarity
 */
public interface ItemSimilarity extends Refreshable {
  
  /**
   * <p>
   * Returns the degree of similarity, of two items, based on the preferences that users have expressed for
   * the items.
   * </p>
   * 
   * @param itemID1 first item ID
   * @param itemID2 second item ID
   * @return similarity between the items, in [-1,1] or {@link Double#NaN} similarity is unknown
   * @throws org.apache.mahout.cf.taste.common.NoSuchItemException
   *  if either item is known to be non-existent in the data
   * @throws TasteException if an error occurs while accessing the data
   */
  double itemSimilarity(long itemID1, long itemID2) throws TasteException;

  /**
   * <p>A bulk-get version of {@link #itemSimilarity(long, long)}.</p>
   *
   * @param itemID1 first item ID
   * @param itemID2s second item IDs to compute similarity with
   * @return similarity between itemID1 and other items
   * @throws org.apache.mahout.cf.taste.common.NoSuchItemException
   *  if any item is known to be non-existent in the data
   * @throws TasteException if an error occurs while accessing the data
   */
  double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException;

  /**
   * @return all IDs of similar items, in no particular order
   */
  long[] allSimilarItemIDs(long itemID) throws TasteException;
}
