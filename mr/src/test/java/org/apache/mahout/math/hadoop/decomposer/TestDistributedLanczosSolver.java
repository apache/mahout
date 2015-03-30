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

package org.apache.mahout.math.hadoop.decomposer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Before;

import java.io.File;
import java.io.IOException;

@Deprecated
public final class TestDistributedLanczosSolver extends MahoutTestCase {

  private int counter = 0;
  private DistributedRowMatrix symCorpus;
  private DistributedRowMatrix asymCorpus;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    File symTestData = getTestTempDir("symTestData");
    File asymTestData = getTestTempDir("asymTestData");
    symCorpus = new TestDistributedRowMatrix().randomDistributedMatrix(100,
        90, 80, 2, 10.0, true, symTestData.getAbsolutePath());
    asymCorpus = new TestDistributedRowMatrix().randomDistributedMatrix(100,
        90, 80, 2, 10.0, false, asymTestData.getAbsolutePath());
  }

  private static String suf(boolean symmetric) {
    return symmetric ? "_sym" : "_asym";
  }

  private DistributedRowMatrix getCorpus(boolean symmetric) {
    return symmetric ? symCorpus : asymCorpus;
  }

  /*
  private LanczosState doTestDistributedLanczosSolver(boolean symmetric,
      int desiredRank) throws IOException {
    return doTestDistributedLanczosSolver(symmetric, desiredRank, true);
  }
   */

  private LanczosState doTestDistributedLanczosSolver(boolean symmetric,
      int desiredRank, boolean hdfsBackedState)
      throws IOException {
    DistributedRowMatrix corpus = getCorpus(symmetric);
    Configuration conf = getConfiguration();
    corpus.setConf(conf);
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    Vector intitialVector = DistributedLanczosSolver.getInitialVector(corpus);
    LanczosState state;
    if (hdfsBackedState) {
      HdfsBackedLanczosState hState = new HdfsBackedLanczosState(corpus,
          desiredRank, intitialVector, new Path(getTestTempDirPath(),
              "lanczosStateDir" + suf(symmetric) + counter));
      hState.setConf(conf);
      state = hState;
    } else {
      state = new LanczosState(corpus, desiredRank, intitialVector);
    }
    solver.solve(state, desiredRank, symmetric);
    SolverTest.assertOrthonormal(state);
    for (int i = 0; i < desiredRank/2; i++) {
      SolverTest.assertEigen(i, state.getRightSingularVector(i), corpus, 0.1, symmetric);
    }
    counter++;
    return state;
  }

  public void doTestResumeIteration(boolean symmetric) throws IOException {
    DistributedRowMatrix corpus = getCorpus(symmetric);
    Configuration conf = getConfiguration();
    corpus.setConf(conf);
    DistributedLanczosSolver solver = new DistributedLanczosSolver();
    int rank = 10;
    Vector intitialVector = DistributedLanczosSolver.getInitialVector(corpus);
    HdfsBackedLanczosState state = new HdfsBackedLanczosState(corpus, rank,
        intitialVector, new Path(getTestTempDirPath(), "lanczosStateDir" + suf(symmetric) + counter));
    solver.solve(state, rank, symmetric);

    rank *= 2;
    state = new HdfsBackedLanczosState(corpus, rank,
        intitialVector, new Path(getTestTempDirPath(), "lanczosStateDir" + suf(symmetric) + counter));
    solver = new DistributedLanczosSolver();
    solver.solve(state, rank, symmetric);

    LanczosState allAtOnceState = doTestDistributedLanczosSolver(symmetric, rank, false);
    for (int i=0; i<state.getIterationNumber(); i++) {
      Vector v = state.getBasisVector(i).normalize();
      Vector w = allAtOnceState.getBasisVector(i).normalize();
      double diff = v.minus(w).norm(2);
      assertTrue("basis " + i + " is too long: " + diff, diff < 0.1);
    }
    counter++;
  }

  // TODO when this can be made to run in under 20 minutes, re-enable
  /*
  @Test
  public void testDistributedLanczosSolver() throws Exception {
    doTestDistributedLanczosSolver(true, 30);
    doTestDistributedLanczosSolver(false, 30);
    doTestResumeIteration(true);
    doTestResumeIteration(false);
  }
   */

}
