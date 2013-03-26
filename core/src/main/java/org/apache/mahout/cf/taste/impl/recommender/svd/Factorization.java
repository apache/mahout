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

import java.util.Arrays;
import java.util.Map;

import com.google.common.base.Preconditions;
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
    this.userIDMapping = Preconditions.checkNotNull(userIDMapping);
    this.itemIDMapping = Preconditions.checkNotNull(itemIDMapping);
    this.userFeatures = userFeatures;
    this.itemFeatures = itemFeatures;
  }

  public double[][] allUserFeatures() {
    return userFeatures;
  }

  public double[] getUserFeatures(long userID) throws NoSuchUserException {
    Integer index = userIDMapping.get(userID);
    if (index == null) {
      throw new NoSuchUserException(userID);
    }
    return userFeatures[index];
  }

  public double[][] allItemFeatures() {
    return itemFeatures;
  }

  public double[] getItemFeatures(long itemID) throws NoSuchItemException {
    Integer index = itemIDMapping.get(itemID);
    if (index == null) {
      throw new NoSuchItemException(itemID);
    }
    return itemFeatures[index];
  }

  public int userIndex(long userID) throws NoSuchUserException {
    Integer index = userIDMapping.get(userID);
    if (index == null) {
      throw new NoSuchUserException(userID);
    }
    return index;
  }

  public Iterable<Map.Entry<Long,Integer>> getUserIDMappings() {
    return userIDMapping.entrySet();
  }

  public int itemIndex(long itemID) throws NoSuchItemException {
    Integer index = itemIDMapping.get(itemID);
    if (index == null) {
      throw new NoSuchItemException(itemID);
    }
    return index;
  }

  public Iterable<Map.Entry<Long,Integer>> getItemIDMappings() {
    return itemIDMapping.entrySet();
  }

  public int numFeatures() {
    return userFeatures.length > 0 ? userFeatures[0].length : 0;
  }

  public int numUsers() {
    return userIDMapping.size();
  }

  public int numItems() {
    return itemIDMapping.size();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof Factorization) {
      Factorization other = (Factorization) o;
      return userIDMapping.equals(other.userIDMapping) && itemIDMapping.equals(other.itemIDMapping)
          && Arrays.deepEquals(userFeatures, other.userFeatures) && Arrays.deepEquals(itemFeatures, other.itemFeatures);
    }
    return false;
  }

  @Override
  public int hashCode() {
    int hashCode = 31 * userIDMapping.hashCode() + itemIDMapping.hashCode();
    hashCode = 31 * hashCode + Arrays.deepHashCode(userFeatures);
    hashCode = 31 * hashCode + Arrays.deepHashCode(itemFeatures);
    return hashCode;
  }
}
