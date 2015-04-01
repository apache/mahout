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

public final class TestChebyshevMeasure extends MahoutTestCase {

  @Test
  public void testMeasure() {

    DistanceMeasure chebyshevDistanceMeasure = new ChebyshevDistanceMeasure();

    Vector[] vectors = {
        new DenseVector(new double[]{1, 0, 0, 0, 0, 0}),
        new DenseVector(new double[]{1, 1, 1, 0, 0, 0}),
        new DenseVector(new double[]{1, 1, 1, 1, 1, 1})
    };
    double[][] distances = {{0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}};
    
    double[][] chebyshevDistanceMatrix = new double[3][3];

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        chebyshevDistanceMatrix[a][b] = chebyshevDistanceMeasure.distance(vectors[a], vectors[b]);
      }
    }
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        assertEquals(distances[a][b], chebyshevDistanceMatrix[a][b], EPSILON);
      }
    }

    assertEquals(0.0, chebyshevDistanceMatrix[0][0], EPSILON);
  }

}
