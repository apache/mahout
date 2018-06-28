/*
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

import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class VectorSimilarityMeasuresTest extends MahoutTestCase {

  static double distributedSimilarity(double[] one,
                                      double[] two,
                                      Class<? extends VectorSimilarityMeasure> similarityMeasureClass) {
    double rand = computeSimilarity(one, two, similarityMeasureClass, new RandomAccessSparseVector(one.length));
    double seq = computeSimilarity(one, two, similarityMeasureClass, new SequentialAccessSparseVector(one.length));
    double dense = computeSimilarity(one, two, similarityMeasureClass, new DenseVector(one.length));
    assertEquals(seq, rand, 1.0e-10);
    assertEquals(seq, dense, 1.0e-10);
    assertEquals(dense, rand, 1.0e-10);
    return seq;
  }

  private static double computeSimilarity(double[] one, double[] two,
      Class<? extends VectorSimilarityMeasure> similarityMeasureClass,
      Vector like) {
    VectorSimilarityMeasure similarityMeasure = ClassUtils.instantiateAs(similarityMeasureClass,
        VectorSimilarityMeasure.class);
    Vector oneNormalized = similarityMeasure.normalize(asVector(one, like));
    Vector twoNormalized = similarityMeasure.normalize(asVector(two, like));

    double normOne = similarityMeasure.norm(oneNormalized);
    double normTwo = similarityMeasure.norm(twoNormalized);

    double dot = 0;
    for (int n = 0; n < one.length; n++) {
      if (oneNormalized.get(n) != 0 && twoNormalized.get(n) != 0) {
        dot += similarityMeasure.aggregate(oneNormalized.get(n), twoNormalized.get(n));
      }
    }

    return similarityMeasure.similarity(dot, normOne, normTwo, one.length);
  }

  static Vector asVector(double[] values, Vector like) {
    Vector vector = like.like();
    for (int dim = 0; dim < values.length; dim++) {
      vector.set(dim, values[dim]);
    }
    return vector;
  }

  @Test
  public void testCooccurrenceCountSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 },
        new double[] { 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, CooccurrenceCountSimilarity.class);

    assertEquals(5.0, similarity, 0);
  }

  @Test
  public void testTanimotoCoefficientSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 },
        new double[] { 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, TanimotoCoefficientSimilarity.class);

    assertEquals(0.454545455, similarity, EPSILON);
  }

  @Test
  public void testCityblockSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 },
        new double[] { 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, CityBlockSimilarity.class);

    assertEquals(0.142857143, similarity, EPSILON);
  }

  @Test
  public void testLoglikelihoodSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 },
        new double[] { 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, LoglikelihoodSimilarity.class);

    assertEquals(0.03320155369284261, similarity, EPSILON);
  }

  @Test
  public void testCosineSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 2, 0, 0, 8, 3, 0, 6, 0, 1, 2, 2, 0 },
        new double[] { 3, 0, 0, 0, 7, 0, 2, 2, 1, 3, 2, 1, 1 }, CosineSimilarity.class);

    assertEquals(0.769846046, similarity, EPSILON);
  }

  @Test
  public void testPearsonCorrelationSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 2, 0, 0, 8, 3, 0, 6, 0, 1, 1, 2, 1 },
        new double[] { 3, 0, 0, 0, 7, 0, 2, 2, 1, 3, 2, 4, 3 }, PearsonCorrelationSimilarity.class);

    assertEquals(0.5303300858899108, similarity, EPSILON);
  }

  @Test
  public void testEuclideanDistanceSimilarity() {
    double similarity = distributedSimilarity(
        new double[] { 0, 2, 0, 0, 8, 3, 0, 6, 0, 1, 1, 2, 1 },
        new double[] { 3, 0, 0, 0, 7, 0, 2, 2, 1, 3, 2, 4, 4 }, EuclideanDistanceSimilarity.class);

    assertEquals(0.11268865367232477, similarity, EPSILON);
  }
}
