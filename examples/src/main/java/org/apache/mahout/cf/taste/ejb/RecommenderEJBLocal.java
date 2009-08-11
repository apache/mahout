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

package org.apache.mahout.cf.taste.ejb;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import javax.ejb.EJBLocalObject;
import java.util.Collection;
import java.util.List;

/**
 * <p>Recommender EJB local component interface.</p>
 *
 * @see RecommenderEJB
 * @see org.apache.mahout.cf.taste.recommender.Recommender
 */
public interface RecommenderEJBLocal extends EJBLocalObject {

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#recommend(long, int)
   */
  List<RecommendedItem> recommend(long userID, int howMany) throws TasteException;

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#recommend(long, int, Rescorer)
   */
  List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer) throws TasteException;

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#estimatePreference(long, long)
   */
  double estimatePreference(long userID, long itemID) throws TasteException;

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#setPreference(long, long, float)
   */
  void setPreference(long userID, long itemID, float value) throws TasteException;

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#removePreference(long, long)
   */
  void removePreference(long userID, long itemID) throws TasteException;

  /**
   * @see org.apache.mahout.cf.taste.recommender.Recommender#refresh(Collection)
   */
  void refresh(Collection<Refreshable> alreadyRefreshed);

}
