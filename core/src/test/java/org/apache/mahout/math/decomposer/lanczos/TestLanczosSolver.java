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

package org.apache.mahout.math.decomposer.lanczos;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.solver.EigenDecomposition;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestLanczosSolver extends SolverTest {
  private static final Logger log = LoggerFactory.getLogger(TestLanczosSolver.class);

  private static final double ERROR_TOLERANCE = 0.05;

  @Test
  public void testEigenvalueCheck() throws Exception {
    int size = 100;
    Matrix m = randomHierarchicalSymmetricMatrix(size);

    Vector initialVector = new DenseVector(size);
    initialVector.assign(1.0 / Math.sqrt(size));
    LanczosSolver solver = new LanczosSolver();
    int desiredRank = 80;
    LanczosState state = new LanczosState(m, desiredRank, initialVector);
    // set initial vector?
    solver.solve(state, desiredRank, true);

    EigenDecomposition decomposition = new EigenDecomposition(m);
    Vector eigenvalues = decomposition.getRealEigenvalues();

    float fractionOfEigensExpectedGood = 0.6f;
    for (int i = 0; i < fractionOfEigensExpectedGood * desiredRank; i++) {
      double s = state.getSingularValue(i);
      double e = eigenvalues.get(i);
      log.info("{} : L = {}, E = {}", i, s, e);
      assertTrue("Singular value differs from eigenvalue", Math.abs((s-e)/e) < ERROR_TOLERANCE);
      Vector v = state.getRightSingularVector(i);
      Vector v2 = decomposition.getV().viewColumn(i);
      double error = 1 - Math.abs(v.dot(v2)/(v.norm(2) * v2.norm(2)));
      log.info("error: {}", error);
      assertTrue(i + ": 1 - cosAngle = " + error, error < ERROR_TOLERANCE);
    }
  }


  @Test
  public void testLanczosSolver() throws Exception {
    int numRows = 800;
    int numColumns = 500;
    Matrix corpus = randomHierarchicalMatrix(numRows, numColumns, false);
    Vector initialVector = new DenseVector(numColumns);
    initialVector.assign(1.0 / Math.sqrt(numColumns));
    int rank = 50;
    LanczosState state = new LanczosState(corpus, rank, initialVector);
    LanczosSolver solver = new LanczosSolver();
    solver.solve(state, rank, false);
    assertOrthonormal(state);
    for (int i = 0; i < rank/2; i++) {
      assertEigen(i, state.getRightSingularVector(i), corpus, ERROR_TOLERANCE, false);
    }
    //assertEigen(eigens, corpus, rank / 2, ERROR_TOLERANCE, false);
  }

  @Test
  public void testLanczosSolverSymmetric() throws Exception {
    int numCols = 500;
    Matrix corpus = randomHierarchicalSymmetricMatrix(numCols);
    Vector initialVector = new DenseVector(numCols);
    initialVector.assign(1.0 / Math.sqrt(numCols));
    int rank = 30;
    LanczosState state = new LanczosState(corpus, rank, initialVector);
    LanczosSolver solver = new LanczosSolver();
    solver.solve(state, rank, true);
    //assertOrthonormal(state);
    //assertEigen(state, rank / 2, ERROR_TOLERANCE, true);
  }

}
