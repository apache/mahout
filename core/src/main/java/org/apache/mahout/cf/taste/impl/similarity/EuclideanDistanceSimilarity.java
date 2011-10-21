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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;

import com.google.common.base.Preconditions;

/**
 * <p>
 * An implementation of a "similarity" based on the Euclidean "distance" between two users X and Y. Thinking
 * of items as dimensions and preferences as points along those dimensions, a distance is computed using all
 * items (dimensions) where both users have expressed a preference for that item. This is simply the square
 * root of the sum of the squares of differences in position (preference) along each dimension.</p>
 * 
 * <p>The similarity could be computed as 1 / (1 + distance), so the resulting values are in the range (0,1].
 * This would weight against pairs that overlap in more dimensions, which should indicate more similarity, 
 * since more dimensions offer more opportunities to be farther apart. Actually, it is computed as 
 * sqrt(n) / (1 + distance), where n is the number of dimensions, in order to help correct for this.
 * sqrt(n) is chosen since randomly-chosen points have a distance that grows as sqrt(n).</p>
 *
 * <p>Note that this could cause a similarity to exceed 1; such values are capped at 1.</p>
 * 
 * <p>Note that the distance isn't normalized in any way; it's not valid to compare similarities computed from
 * different domains (different rating scales, for example). Within one domain, normalizing doesn't matter much as
 * it doesn't change ordering.</p>
 */
public final class EuclideanDistanceSimilarity extends AbstractSimilarity {

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public EuclideanDistanceSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public EuclideanDistanceSimilarity(DataModel dataModel, Weighting weighting) throws TasteException {
    super(dataModel, weighting, false);
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }
  
  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    return 1.0 / (1.0 + Math.sqrt(sumXYdiff2) / Math.sqrt(n));
  }
  
}
