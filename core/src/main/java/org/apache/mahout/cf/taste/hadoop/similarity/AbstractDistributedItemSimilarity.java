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
 * abstract base class for all implementations of {@link DistributedItemSimilarity} that does not give a
 * weight to item vectors and only ensures that the result is within [-1,1]
 */
public abstract class AbstractDistributedItemSimilarity
    implements DistributedItemSimilarity {

  @Override
  public final double similarity(Iterable<CoRating> coratings,
                                 double weightOfItemVectorX,
                                 double weightOfItemVectorY,
                                 int numberOfUsers) {

    double result = doComputeResult(coratings, weightOfItemVectorX, weightOfItemVectorY, numberOfUsers);

    if (result < -1.0) {
      result = -1.0;
    } else if (result > 1.0) {
      result = 1.0;
    }
    return result;
  }

  /**
   * do not compute a weight by default, subclasses can override this
   * when they need a weight
   */
  @Override
  public double weightOfItemVector(Iterable<Float> prefValues) {
    return Double.NaN;
  }

  protected abstract double doComputeResult(Iterable<CoRating> coratings,
      double weightOfItemVectorX, double weightOfItemVectorY,
      int numberOfUsers);
}
