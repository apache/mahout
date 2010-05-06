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

public final class DistributedUncenteredZeroAssumingCosineSimilarity
    implements DistributedSimilarity {

  @Override
  public double similarity(Iterator<CoRating> coRatings, double weightOfItemVectorX, double weightOfItemVectorY) {

    double sumXY = 0;
    while (coRatings.hasNext()) {
      CoRating coRating = coRatings.next();
      sumXY += coRating.getPrefValueX() * coRating.getPrefValueY();
    }

    if (sumXY == 0) {
      return Double.NaN;
    }
    return sumXY / (weightOfItemVectorX * weightOfItemVectorY);
  }

  @Override
  public double weightOfItemVector(Iterator<Float> prefValues) {
    double length = 0.0;
    while (prefValues.hasNext()) {
      float prefValue = prefValues.next();
      if (!((Float)prefValue).isNaN()) {
        length += prefValue * prefValue;
      }
    }

    return Math.sqrt(length);
  }

}
