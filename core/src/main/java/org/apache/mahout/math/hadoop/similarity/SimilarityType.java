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

package org.apache.mahout.math.hadoop.similarity;

import java.util.Arrays;

import org.apache.mahout.math.hadoop.similarity.vector.DistributedCityBlockVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedCooccurrenceVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedEuclideanDistanceVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedLoglikelihoodVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedPearsonCorrelationVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedTanimotoCoefficientVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedUncenteredCosineVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedUncenteredZeroAssumingCosineVectorSimilarity;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedVectorSimilarity;

public enum SimilarityType {

  SIMILARITY_COOCCURRENCE(DistributedCooccurrenceVectorSimilarity.class),
  SIMILARITY_EUCLIDEAN_DISTANCE(DistributedEuclideanDistanceVectorSimilarity.class),
  SIMILARITY_LOGLIKELIHOOD(DistributedLoglikelihoodVectorSimilarity.class),
  SIMILARITY_PEARSON_CORRELATION(DistributedPearsonCorrelationVectorSimilarity.class),
  SIMILARITY_TANIMOTO_COEFFICIENT(DistributedTanimotoCoefficientVectorSimilarity.class),
  SIMILARITY_UNCENTERED_COSINE(DistributedUncenteredCosineVectorSimilarity.class),
  SIMILARITY_UNCENTERED_ZERO_ASSUMING_COSINE(DistributedUncenteredZeroAssumingCosineVectorSimilarity.class),
  SIMILARITY_CITY_BLOCK(DistributedCityBlockVectorSimilarity.class);

  private final Class<? extends DistributedVectorSimilarity> similarityImplementation;

  SimilarityType(Class<? extends DistributedVectorSimilarity> similarityImplementation) {
    this.similarityImplementation = similarityImplementation;
  }

  public String getSimilarityImplementationClassName() {
    return similarityImplementation.getName();
  }

  public static String listEnumNames() {
    return Arrays.toString(values());
  }

}
