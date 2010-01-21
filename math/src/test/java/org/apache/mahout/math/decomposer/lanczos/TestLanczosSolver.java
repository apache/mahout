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
import org.apache.mahout.math.decomposer.SolverTest;

import java.util.ArrayList;
import java.util.List;

public class TestLanczosSolver extends SolverTest {

  public TestLanczosSolver(String name) {
    super(name);
  }

  public void testLanczosSolver() throws Exception {
    int numColumns = 800;
    Matrix corpus = randomSequentialAccessSparseMatrix(1000, 900, numColumns, 30, 1.0);
    int rank = 50;
    Matrix eigens = new DenseMatrix(rank, numColumns);
    long time = timeLanczos(corpus, eigens, rank);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    assertOrthonormal(eigens);
    assertEigen(eigens, corpus, 0.1);
  }

  public static long timeLanczos(Matrix corpus, Matrix eigens, int rank) throws Exception {
    long start = System.currentTimeMillis();

    LanczosSolver solver = new LanczosSolver();
    List<Double> eVals = new ArrayList<Double>();
    solver.solve(corpus, rank, eigens, eVals);
    
    long end = System.currentTimeMillis();
    return end - start;
  }

}
