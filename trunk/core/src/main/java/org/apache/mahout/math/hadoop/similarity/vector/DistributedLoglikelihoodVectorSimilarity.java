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

package org.apache.mahout.math.hadoop.similarity.vector;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.similarity.Cooccurrence;
import org.apache.mahout.math.stats.LogLikelihood;

/**
 * distributed implementation of loglikelihood as vector similarity measure
 */
public class DistributedLoglikelihoodVectorSimilarity extends
    AbstractDistributedVectorSimilarity {

  @Override
  protected double doComputeResult(int rowA, int rowB, Iterable<Cooccurrence> cooccurrences, double weightOfVectorA,
      double weightOfVectorB, int numberOfColumns) {

    int cooccurrenceCount = countElements(cooccurrences);
    if (cooccurrenceCount == 0) {
      return Double.NaN;
    }

    int occurrencesA = (int) weightOfVectorA;
    int occurrencesB = (int) weightOfVectorB;

    double logLikelihood =
        LogLikelihood.logLikelihoodRatio(cooccurrenceCount,
                                         occurrencesB - cooccurrenceCount,
                                         occurrencesA - cooccurrenceCount,
                                         numberOfColumns - occurrencesA - occurrencesB + cooccurrenceCount);

    return 1.0 - 1.0 / (1.0 + logLikelihood);
  }

  @Override
  public double weight(Vector v) {
    return (double) countElements(v.iterateNonZero());
  }

}
