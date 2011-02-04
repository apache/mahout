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

package org.apache.mahout.common.distance;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public final class TestMinkowskiMeasure extends MahoutTestCase {

  @Test
  public void testMeasure() {

    DistanceMeasure minkowskiDistanceMeasure = new MinkowskiDistanceMeasure(1.5);
    DistanceMeasure manhattanDistanceMeasure = new ManhattanDistanceMeasure();
    DistanceMeasure euclideanDistanceMeasure = new EuclideanDistanceMeasure();

    Vector[] vectors = {
        new DenseVector(new double[]{1, 0, 0, 0, 0, 0}),
        new DenseVector(new double[]{1, 1, 1, 0, 0, 0}),
        new DenseVector(new double[]{1, 1, 1, 1, 1, 1})
    };

    double[][] minkowskiDistanceMatrix = new double[3][3];
    double[][] manhattanDistanceMatrix = new double[3][3];
    double[][] euclideanDistanceMatrix = new double[3][3];

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        minkowskiDistanceMatrix[a][b] = minkowskiDistanceMeasure.distance(vectors[a], vectors[b]);
        manhattanDistanceMatrix[a][b] = manhattanDistanceMeasure.distance(vectors[a], vectors[b]);
        euclideanDistanceMatrix[a][b] = euclideanDistanceMeasure.distance(vectors[a], vectors[b]);
      }
    }

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        assertTrue(minkowskiDistanceMatrix[a][b] <= manhattanDistanceMatrix[a][b]);
        assertTrue(minkowskiDistanceMatrix[a][b] >= euclideanDistanceMatrix[a][b]);
      }
    }

    assertEquals(0.0, minkowskiDistanceMatrix[0][0], EPSILON);
    assertTrue(minkowskiDistanceMatrix[0][0] < minkowskiDistanceMatrix[0][1]);
    assertTrue(minkowskiDistanceMatrix[0][1] < minkowskiDistanceMatrix[0][2]);
  }

}
