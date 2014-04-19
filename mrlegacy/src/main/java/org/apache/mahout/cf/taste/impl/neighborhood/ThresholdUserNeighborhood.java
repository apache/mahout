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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.SamplingLongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Computes a neigbhorhood consisting of all users whose similarity to the given user meets or exceeds a
 * certain threshold. Similarity is defined by the given {@link UserSimilarity}.
 * </p>
 */
public final class ThresholdUserNeighborhood extends AbstractUserNeighborhood {
  
  private final double threshold;
  
  /**
   * @param threshold
   *          similarity threshold
   * @param userSimilarity
   *          similarity metric
   * @param dataModel
   *          data model
   * @throws IllegalArgumentException
   *           if threshold is {@link Double#NaN}, or if samplingRate is not positive and less than or equal
   *           to 1.0, or if userSimilarity or dataModel are {@code null}
   */
  public ThresholdUserNeighborhood(double threshold, UserSimilarity userSimilarity, DataModel dataModel) {
    this(threshold, userSimilarity, dataModel, 1.0);
  }
  
  /**
   * @param threshold
   *          similarity threshold
   * @param userSimilarity
   *          similarity metric
   * @param dataModel
   *          data model
   * @param samplingRate
   *          percentage of users to consider when building neighborhood -- decrease to trade quality for
   *          performance
   * @throws IllegalArgumentException
   *           if threshold or samplingRate is {@link Double#NaN}, or if samplingRate is not positive and less
   *           than or equal to 1.0, or if userSimilarity or dataModel are {@code null}
   */
  public ThresholdUserNeighborhood(double threshold,
                                   UserSimilarity userSimilarity,
                                   DataModel dataModel,
                                   double samplingRate) {
    super(userSimilarity, dataModel, samplingRate);
    Preconditions.checkArgument(!Double.isNaN(threshold), "threshold must not be NaN");
    this.threshold = threshold;
  }
  
  @Override
  public long[] getUserNeighborhood(long userID) throws TasteException {
    
    DataModel dataModel = getDataModel();
    FastIDSet neighborhood = new FastIDSet();
    LongPrimitiveIterator usersIterable = SamplingLongPrimitiveIterator.maybeWrapIterator(dataModel
        .getUserIDs(), getSamplingRate());
    UserSimilarity userSimilarityImpl = getUserSimilarity();
    
    while (usersIterable.hasNext()) {
      long otherUserID = usersIterable.next();
      if (userID != otherUserID) {
        double theSimilarity = userSimilarityImpl.userSimilarity(userID, otherUserID);
        if (!Double.isNaN(theSimilarity) && theSimilarity >= threshold) {
          neighborhood.add(otherUserID);
        }
      }
    }
    
    return neighborhood.toArray();
  }
  
  @Override
  public String toString() {
    return "ThresholdUserNeighborhood";
  }
  
}
