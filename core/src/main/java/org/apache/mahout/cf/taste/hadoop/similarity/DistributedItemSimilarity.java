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

package org.apache.mahout.cf.taste.hadoop.similarity;

import java.util.Iterator;

/**
 * Modelling the pairwise similarity computation of items in a distributed manner
 */
public interface DistributedItemSimilarity {

  /**
   * compute the weight of an item vector (called in an early stage of the map-reduce steps)
   *
   * @param prefValues
   * @return
   */
  double weightOfItemVector(Iterator<Float> prefValues);

  /**
   * compute the similarity for a pair of item-vectors
   *
   * @param coratings all coratings for these items
   * @param weightOfItemVectorX the weight computed for the first vector
   * @param weightOfItemVectorY the weight computed for the second vector
   * @param numberOfUsers the overall number of users
   * @return
   */
  double similarity(Iterator<CoRating> coratings,
                    double weightOfItemVectorX,
                    double weightOfItemVectorY,
                    int numberOfUsers);
}
