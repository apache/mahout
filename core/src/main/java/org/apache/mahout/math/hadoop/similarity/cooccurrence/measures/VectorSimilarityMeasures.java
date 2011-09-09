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

package org.apache.mahout.math.hadoop.similarity.cooccurrence.measures;

import java.util.Arrays;

public enum VectorSimilarityMeasures {

  SIMILARITY_COOCCURRENCE(CooccurrenceCountSimilarity.class),
  SIMILARITY_LOGLIKELIHOOD(LoglikelihoodSimilarity.class),
  SIMILARITY_TANIMOTO_COEFFICIENT(TanimotoCoefficientSimilarity.class),
  SIMILARITY_CITY_BLOCK(CityBlockSimilarity.class),
  SIMILARITY_COSINE(CosineSimilarity.class),
  SIMILARITY_PEARSON_CORRELATION(PearsonCorrelationSimilarity.class),
  SIMILARITY_EUCLIDEAN_DISTANCE(EuclideanDistanceSimilarity.class);

  private final Class<? extends VectorSimilarityMeasure> implementingClass;

  VectorSimilarityMeasures(Class<? extends VectorSimilarityMeasure> impl) {
    this.implementingClass = impl;
  }

  public String getClassname() {
    return implementingClass.getName();
  }

  public static String list() {
    return Arrays.toString(values());
  }

}
