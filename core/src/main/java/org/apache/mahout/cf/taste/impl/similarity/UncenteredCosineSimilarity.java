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
 * An implementation of the cosine similarity. The result is the cosine of the angle formed between
 * the two preference vectors.
 * </p>
 *
 * <p>
 * Note that this similarity does not "center" its data, shifts the user's preference values so that each of their
 * means is 0. For this behavior, use {@link PearsonCorrelationSimilarity}, which actually is mathematically
 * equivalent for centered data.
 * </p>
 */
public final class UncenteredCosineSimilarity extends AbstractSimilarity {

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public UncenteredCosineSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  /**
   * @throws IllegalArgumentException if {@link DataModel} does not have preference values
   */
  public UncenteredCosineSimilarity(DataModel dataModel, Weighting weighting) throws TasteException {
    super(dataModel, weighting, false);
    Preconditions.checkArgument(dataModel.hasPreferenceValues(), "DataModel doesn't have preference values");
  }

  @Override
  double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2) {
    if (n == 0) {
      return Double.NaN;
    }
    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much similarity under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }

}
