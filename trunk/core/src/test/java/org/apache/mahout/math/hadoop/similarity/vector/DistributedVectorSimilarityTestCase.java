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

import java.util.Collection;
import java.util.LinkedList;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.similarity.Cooccurrence;
import org.apache.mahout.math.hadoop.similarity.RowSimilarityJob;

/**
 * base testcase for all tests of classes implementing {@link DistributedVectorSimilarity}
 */
public abstract class DistributedVectorSimilarityTestCase extends MahoutTestCase {

  /**
   * convenience method to create a {@link Vector}
   */
  static Vector asVector(double... values) {
    return new DenseVector(values);
  }

  /**
   * @see DistributedVectorSimilarityTestCase#assertSimilar(DistributedVectorSimilarity, int, int, Vector, Vector,
   * int, double)
   */
  static void assertSimilar(DistributedVectorSimilarity similarity, Vector v1, Vector v2, int numberOfColumns,
      double expectedSimilarity) {
    assertSimilar(similarity, 1, 2, v1, v2, numberOfColumns, expectedSimilarity);
  }

  /**
   * emulates the way similarities are computed by {@link RowSimilarityJob}
   */
  static void assertSimilar(DistributedVectorSimilarity similarity, int rowA, int rowB, Vector v1, Vector v2,
      int numberOfColumns, double expectedSimilarity) {

    double weightA = similarity.weight(v1);
    double weightB = similarity.weight(v2);

    Collection<Cooccurrence> cooccurrences = new LinkedList<Cooccurrence>();
    for (int n = 0; n < numberOfColumns; n++) {
      double valueA = v1.get(n);
      double valueB = v2.get(n);
      if (valueA != 0.0 && valueB != 0.0) {
        cooccurrences.add(new Cooccurrence(n, valueA, valueB));
      }
    }

    double result = similarity.similarity(rowA, rowB, cooccurrences, weightA, weightB, numberOfColumns);
    if (Double.isNaN(expectedSimilarity)) {
      assertTrue(Double.isNaN(result));
    } else {
      assertEquals(expectedSimilarity, result, EPSILON);
    }
  }
}
