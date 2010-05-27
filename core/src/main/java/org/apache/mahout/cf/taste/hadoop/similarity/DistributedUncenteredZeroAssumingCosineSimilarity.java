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

/**
 * distributed cosine similarity that assumes that all unknown preferences
 * are zeros and that does not center data
 */
public final class DistributedUncenteredZeroAssumingCosineSimilarity
    extends AbstractDistributedItemSimilarity {

  @Override
  protected double doComputeResult(Iterable<CoRating> coRatings,
                                   double weightOfItemVectorX,
                                   double weightOfItemVectorY,
                                   int numberOfUsers) {

    double sumXY = 0.0;
    for (CoRating coRating : coRatings) {
      sumXY += coRating.getPrefValueX() * coRating.getPrefValueY();
    }

    if (sumXY == 0.0) {
      return Double.NaN;
    }
    return sumXY / (weightOfItemVectorX * weightOfItemVectorY);
  }

  @Override
  public double weightOfItemVector(Iterable<Float> prefValues) {
    double length = 0.0;
    for (float prefValue : prefValues) {
      if (!Float.isNaN(prefValue)) {
        length += prefValue * prefValue;
      }
    }

    return Math.sqrt(length);
  }

}
