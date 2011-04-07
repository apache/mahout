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

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.linalg.EigenvalueDecomposition;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public final class TestLanczosSolver extends SolverTest {
  private static final Logger log = LoggerFactory.getLogger(TestLanczosSolver.class);

  private static final double ERROR_TOLERANCE = 1.0e-5;

  @Test
  public void testEigenvalueCheck() throws Exception {
    int size = 100;
    Matrix m = randomHierarchicalSymmetricMatrix(size);
    int desiredRank = 80;
    LanczosSolver solver = new LanczosSolver();
    Matrix eigenvectors = new DenseMatrix(desiredRank, size);
    List<Double> eigenvalueList = new ArrayList<Double>();
    solver.solve(m, desiredRank, eigenvectors, eigenvalueList);

    EigenvalueDecomposition decomposition = new EigenvalueDecomposition(m);
    DoubleMatrix1D eigenvalues = decomposition.getRealEigenvalues();

    float fractionOfEigensExpectedGood = 0.75f;
    for(int i = 0; i < fractionOfEigensExpectedGood * desiredRank; i++) {
      log.info(i + " : L = {}, E = {}",
          eigenvalueList.get(desiredRank - i - 1),
          eigenvalues.get(eigenvalues.size() - i - 1) );
      Vector v = eigenvectors.getRow(i);
      Vector v2 = decomposition.getV().viewColumn(eigenvalues.size() - i - 1).toVector();
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
    int rank = 50;
    Matrix eigens = new DenseMatrix(rank, numColumns);
    long time = timeLanczos(corpus, eigens, rank, false);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    assertOrthonormal(eigens);
    assertEigen(eigens, corpus, rank / 2, ERROR_TOLERANCE, false);
  }

  @Test
  public void testLanczosSolverSymmetric() throws Exception {
    Matrix corpus = randomHierarchicalSymmetricMatrix(500);
    int rank = 30;
    Matrix eigens = new DenseMatrix(rank, corpus.numCols());
    long time = timeLanczos(corpus, eigens, rank, true);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    assertOrthonormal(eigens);
    assertEigen(eigens, corpus, rank / 2, ERROR_TOLERANCE, true);
  }

  public static long timeLanczos(Matrix corpus, Matrix eigens, int rank, boolean symmetric) {
    long start = System.currentTimeMillis();

    LanczosSolver solver = new LanczosSolver();
    List<Double> eVals = new ArrayList<Double>();
    solver.solve(corpus, rank, eigens, eVals, symmetric);
    
    long end = System.currentTimeMillis();
    return end - start;
  }

}
