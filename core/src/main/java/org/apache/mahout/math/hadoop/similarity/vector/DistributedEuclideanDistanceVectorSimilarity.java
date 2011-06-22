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

import org.apache.mahout.math.hadoop.similarity.Cooccurrence;

/**
 * distributed implementation of euclidean distance as vector similarity measure
 */
public class DistributedEuclideanDistanceVectorSimilarity extends AbstractDistributedVectorSimilarity {

  @Override
  protected double doComputeResult(int rowA, int rowB, Iterable<Cooccurrence> cooccurrences, double weightOfVectorA,
      double weightOfVectorB, long numberOfColumns) {

    double n = 0.0;
    double sumXYdiff2 = 0.0;

    for (Cooccurrence cooccurrence : cooccurrences) {
      double diff = cooccurrence.getValueA() - cooccurrence.getValueB();
      sumXYdiff2 += diff * diff;
      n++;
    }

    return n / (1.0 + Math.sqrt(sumXYdiff2));
  }

}
