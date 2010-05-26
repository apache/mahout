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

import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;

/**
 * Distributed version of {@link PearsonCorrelationSimilarity}
 */
public class DistributedPearsonCorrelationSimilarity extends AbstractDistributedItemSimilarity {

  @Override
  protected double doComputeResult(Iterable<CoRating> coRatings,
                                   double weightOfItemVectorX,
                                   double weightOfItemVectorY,
                                   int numberOfUsers) {

    int count = 0;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;

    for (CoRating coRating : coRatings) {
      double x = coRating.getPrefValueX();
      double y = coRating.getPrefValueY();

      sumXY += x * y;
      sumX += x;
      sumX2 += x * x;
      sumY += y;
      sumY2 += y * y;
      count++;
    }

    if (sumXY == 0.0) {
      return Double.NaN;
    }

    // "Center" the data. If my math is correct, this'll do it.
    double n = count;
    double meanX = sumX / n;
    double meanY = sumY / n;
    // double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
    double centeredSumXY = sumXY - meanY * sumX;
    // double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
    double centeredSumX2 = sumX2 - meanX * sumX;
    // double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;
    double centeredSumY2 = sumY2 - meanY * sumY;

    double denominator = Math.sqrt(centeredSumX2) * Math.sqrt(centeredSumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }

    return centeredSumXY / denominator;
  }
}
