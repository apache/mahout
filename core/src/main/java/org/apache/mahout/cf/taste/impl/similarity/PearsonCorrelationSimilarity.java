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
 * An implementation of the Pearson correlation. For users X and Y, the following values are calculated:
 * </p>
 *
 * <ul>
 * <li>sumX2: sum of the square of all X's preference values</li>
 * <li>sumY2: sum of the square of all Y's preference values</li>
 * <li>sumXY: sum of the product of X and Y's preference value for all items for which both X and Y express a
 * preference</li>
 * </ul>
 *
 * <p>
 * The correlation is then:
 *
 * <p>
 * {@code sumXY / sqrt(sumX2 * sumY2)}
 * </p>
 *
 * <p>
 * Note that this correlation "centers" its data, shifts the user's preference values so that each of their
 * means is 0. This is necessary to achieve expected behavior on all data sets.
 * </p>
 *
 * <p>
 * This correlation implementation is equivalent to the cosine similarity since the data it receives
 * is assumed to be centered -- mean is 0. The correlation may be interpreted as the cosine of the angle
 * between the two vectors defined by the users' preference values.
 * </p>
 *
 * <p>
 * For cosine similarity on uncentered data, see {@link UncenteredCosineSimilarity}.
 * </p> 
 */
public final class PearsonCorrelationSimilarity extends AbstractSimilarity {

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public PearsonCorrelationSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public PearsonCorrelationSimilarity(DataModel dataModel, Weighting weighting) throws TasteException {
    super(dataModel, weighting, true);
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }
  
  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    if (n == 0) {
      return Double.NaN;
    }
    // Note that sum of X and sum of Y don't appear here since they are assumed to be 0;
    // the data is assumed to be centered.
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }
  
}
