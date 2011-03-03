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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;

/**
 * a factorization of the rating matrix
 */
public class Factorization {

  /** used to find the rows in the user features matrix by userID */
  private final FastByIDMap<Integer> userIDMapping;
  /** used to find the rows in the item features matrix by itemID */
  private final FastByIDMap<Integer> itemIDMapping;

  /** user features matrix */
  private final double[][] userFeatures;
  /** item features matrix */
  private final double[][] itemFeatures;

  public Factorization(FastByIDMap<Integer> userIDMapping, FastByIDMap<Integer> itemIDMapping, double[][] userFeatures,
      double[][] itemFeatures) {
    this.userIDMapping = userIDMapping;
    this.itemIDMapping = itemIDMapping;
    this.userFeatures = userFeatures;
    this.itemFeatures = itemFeatures;
  }

  public double[] getUserFeatures(long userID) throws NoSuchUserException {
    Integer index = userIDMapping.get(userID);
    if (index == null) {
      throw new NoSuchUserException(userID);
    }
    return userFeatures[index];
  }

  public double[] getItemFeatures(long itemID) throws NoSuchItemException {
    Integer index = itemIDMapping.get(itemID);
    if (index == null) {
      throw new NoSuchItemException(itemID);
    }
    return itemFeatures[index];
  }

}
