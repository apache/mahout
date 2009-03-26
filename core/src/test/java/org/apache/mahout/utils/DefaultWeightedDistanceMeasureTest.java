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

package org.apache.mahout.utils;

import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

public abstract class DefaultWeightedDistanceMeasureTest extends DefaultDistanceMeasureTest {

  @Override
  public abstract WeightedDistanceMeasure distanceMeasureFactory();

  public void testMeasureWeighted() {

    WeightedDistanceMeasure distanceMeasure = distanceMeasureFactory();

    Vector[] vectors = {
        new DenseVector(new double[]{9, 9, 1}),
        new DenseVector(new double[]{1, 9, 9}),
        new DenseVector(new double[]{9, 1, 9}),
    };
    distanceMeasure.setWeights(new DenseVector(new double[]{1, 1000, 1}));

    double[][] distanceMatrix = new double[3][3];

    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        distanceMatrix[a][b] = distanceMeasure.distance(vectors[a], vectors[b]);
      }
    }

    assertEquals(0.0, distanceMatrix[0][0]);
    assertTrue(distanceMatrix[0][1] < distanceMatrix[0][2]);


  }


}
