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

import org.apache.mahout.cf.taste.impl.common.RandomUtils;

/** Simply encapsulates a user and a similarity value. */
public final class SimilarUser implements Comparable<SimilarUser> {

  private final long userID;
  private final double similarity;

  public SimilarUser(long userID, double similarity) {
    this.userID = userID;
    this.similarity = similarity;
  }

  long getUserID() {
    return userID;
  }

  double getSimilarity() {
    return similarity;
  }

  @Override
  public int hashCode() {
    return (int) userID ^ RandomUtils.hashDouble(similarity);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof SimilarUser)) {
      return false;
    }
    SimilarUser other = (SimilarUser) o;
    return userID == other.userID && similarity == other.similarity;
  }

  @Override
  public String toString() {
    return "SimilarUser[user:" + userID + ", similarity:" + similarity + ']';
  }

  /** Defines an ordering from most similar to least similar. */
  @Override
  public int compareTo(SimilarUser other) {
    double otherSimilarity = other.similarity;
    return similarity > otherSimilarity ? -1 : similarity < otherSimilarity ? 1 : 0;
  }

}
