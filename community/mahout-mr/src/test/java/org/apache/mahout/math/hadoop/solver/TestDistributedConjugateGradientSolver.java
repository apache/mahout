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

package org.apache.mahout.math.hadoop.solver;

import java.io.File;
import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;

public final class TestDistributedConjugateGradientSolver extends MahoutTestCase {

  private static Vector randomVector(int size, double entryMean) {
    DenseVector v = new DenseVector(size);
    Random r = RandomUtils.getRandom();
    
    for (int i = 0; i < size; ++i) {
      v.setQuick(i, r.nextGaussian() * entryMean);
    }
    
    return v;
  }

  @Test
  public void testSolver() throws Exception {
    File testData = getTestTempDir("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        10, 10, 10, 10, 10.0, true, testData.getAbsolutePath());
    matrix.setConf(getConfiguration());
    Vector vector = randomVector(matrix.numCols(), 10.0);
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    Vector x = solver.solve(matrix, vector);

    Vector solvedVector = matrix.times(x);    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }
}
