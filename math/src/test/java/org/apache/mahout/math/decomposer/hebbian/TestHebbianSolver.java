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

package org.apache.mahout.math.decomposer.hebbian;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import org.apache.mahout.math.decomposer.AsyncEigenVerifier;
import org.apache.mahout.math.decomposer.SolverTest;
import org.junit.Test;

/**
 * This test is woefully inadequate, and also requires tons of memory, because it's part
 * unit test, part performance test, and part comparison test (between the Hebbian and Lanczos
 * approaches).
 * TODO: make better.
 */
public final class TestHebbianSolver extends SolverTest {

  public static long timeSolver(Matrix corpus,
                                double convergence,
                                int maxNumPasses,
                                TrainingState state) {
    return timeSolver(corpus,
                      convergence,
                      maxNumPasses,
                      10,
                      state);
  }

  public static long timeSolver(Matrix corpus,
                                double convergence,
                                int maxNumPasses,
                                int desiredRank,
                                TrainingState state) {
    HebbianUpdater updater = new HebbianUpdater();
    AsyncEigenVerifier verifier = new AsyncEigenVerifier();
    HebbianSolver solver = new HebbianSolver(updater,
                                             verifier,
                                             convergence,
                                             maxNumPasses);
    long start = System.nanoTime();
    TrainingState finalState = solver.solve(corpus, desiredRank);
    assertNotNull(finalState);
    state.setCurrentEigens(finalState.getCurrentEigens());
    state.setCurrentEigenValues(finalState.getCurrentEigenValues());
    long time = 0L;
    time += System.nanoTime() - start;
    verifier.close();
    assertEquals(state.getCurrentEigens().numRows(), desiredRank);
    return time / 1000000L;
  }



  public static long timeSolver(Matrix corpus, TrainingState state) {
    return timeSolver(corpus, state, 10);
  }

  public static long timeSolver(Matrix corpus, TrainingState state, int rank) {
    return timeSolver(corpus, 0.01, 20, rank, state);
  }

  @Test
  public void testHebbianSolver() {
    int numColumns = 800;
    Matrix corpus = randomSequentialAccessSparseMatrix(1000, 900, numColumns, 30, 1.0);
    int rank = 50;
    Matrix eigens = new DenseMatrix(rank, numColumns);
    TrainingState state = new TrainingState(eigens, null);
    long optimizedTime = timeSolver(corpus,
                                    0.00001,
                                    5,
                                    rank,
                                    state);
    eigens = state.getCurrentEigens();
    assertEigen(eigens, corpus, 0.05, false);
    assertOrthonormal(eigens, 1.0e-6);
    System.out.println("Avg solving (Hebbian) time in ms: " + optimizedTime);
  }

  /*
  public void testSolverWithSerialization() throws Exception
  {
    _corpusProjectionsVectorFactory = new DenseMapVectorFactory();
    _eigensVectorFactory = new DenseMapVectorFactory();
    
    timeSolver(TMP_EIGEN_DIR,
               0.001, 
               5, 
               new TrainingState(null, null));
    
    File eigenDir = new File(TMP_EIGEN_DIR + File.separator + HebbianSolver.EIGEN_VECT_DIR);
    DiskBufferedDoubleMatrix eigens = new DiskBufferedDoubleMatrix(eigenDir, 10);
    
    DoubleMatrix inMemoryMatrix = new HashMapDoubleMatrix(_corpusProjectionsVectorFactory, eigens);
    
    for (Entry<Integer, MapVector> diskEntry : eigens)
    {
      for (Entry<Integer, MapVector> inMemoryEntry : inMemoryMatrix)
      {
        if (diskEntry.getKey() - inMemoryEntry.getKey() == 0)
        {
          assertTrue("vector with index : " + diskEntry.getKey() + " is not the same on disk as in memory", 
                     Math.abs(1 - diskEntry.getValue().dot(inMemoryEntry.getValue())) < 1e-6);
        }
        else
        {
          assertTrue("vector with index : " + diskEntry.getKey() 
                     + " is not orthogonal to memory vect with index : " + inMemoryEntry.getKey(),
                     Math.abs(diskEntry.getValue().dot(inMemoryEntry.getValue())) < 1e-6);
        }
      }
    }
    File corpusDir = new File(TMP_EIGEN_DIR + File.separator + "corpus");
    corpusDir.mkdir();
    // TODO: persist to disk?
   // DiskBufferedDoubleMatrix.persistChunk(corpusDir, corpus, true);
   // eigens.delete();
    
   // DiskBufferedDoubleMatrix.delete(new File(TMP_EIGEN_DIR));
  }
  */
/*
  public void testHebbianVersusLanczos() throws Exception
  {
    _corpusProjectionsVectorFactory = new DenseMapVectorFactory();
    _eigensVectorFactory = new DenseMapVectorFactory();
    int desiredRank = 200;
    long time = timeSolver(TMP_EIGEN_DIR,
                           0.00001,
                           5, 
                           desiredRank,
                           new TrainingState());

    System.out.println("Hebbian time: " + time + "ms");
    File eigenDir = new File(TMP_EIGEN_DIR + File.separator + HebbianSolver.EIGEN_VECT_DIR);
    DiskBufferedDoubleMatrix eigens = new DiskBufferedDoubleMatrix(eigenDir, 10);
    
    DoubleMatrix2D srm = asSparseDoubleMatrix2D(corpus);
    long timeA = System.nanoTime();
    EigenvalueDecomposition asSparseRealDecomp = new EigenvalueDecomposition(srm);
    for (int i=0; i<desiredRank; i++)
      asSparseRealDecomp.getEigenvector(i);
    System.out.println("CommonsMath time: " + (System.nanoTime() - timeA)/TimingConstants.NANOS_IN_MILLI + "ms");
    
   // System.out.println("Hebbian results:");
   // printEigenVerify(eigens, corpus);
    
    DoubleMatrix lanczosEigenVectors = new HashMapDoubleMatrix(new HashMapVectorFactory());
    List<Double> lanczosEigenValues = new ArrayList<Double>();
 
    LanczosSolver solver = new LanczosSolver();
    solver.solve(corpus, desiredRank*5, lanczosEigenVectors, lanczosEigenValues);
    
    for (TimingSection section : LanczosSolver.TimingSection.values())
    {
      System.out.println("Lanczos " + section.toString() + " = " + (int)(solver.getTimeMillis(section)/1000) + " seconds");
    }
    
   // System.out.println("\nLanczos results:");
   // printEigenVerify(lanczosEigenVectors, corpus);
  }
  
  private DoubleMatrix2D asSparseDoubleMatrix2D(Matrix corpus)
  {
    DoubleMatrix2D result = new DenseDoubleMatrix2D(corpus.numRows(), corpus.numRows());
    for (int i=0; i<corpus.numRows(); i++) {
      for (int j=i; j<corpus.numRows(); j++) {
        double v = corpus.getRow(i).dot(corpus.getRow(j));
        result.set(i, j, v);
        result.set(j, i, v);
      }
    }
    return result;
  }


  public static void printEigenVerify(DoubleMatrix eigens, DoubleMatrix corpus)
  {
    for (Map.Entry<Integer, MapVector> entry : eigens)
    {
      MapVector eigen = entry.getValue();
      MapVector afterMultiply = corpus.timesSquared(eigen);
      double norm = afterMultiply.norm();
      double error = 1 - eigen.dot(afterMultiply) / (eigen.norm() * afterMultiply.norm());
      System.out.println(entry.getKey() + ": error = " + error + ", eVal = " + (norm / eigen.norm()));
    }
  }
    */

}
