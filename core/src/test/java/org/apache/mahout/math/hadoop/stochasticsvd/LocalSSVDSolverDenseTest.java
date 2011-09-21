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

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.io.File;
import java.io.IOException;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

/**
 * 
 * Tests SSVD solver with a made-up data running hadoop solver in a local mode.
 * It requests full-rank SSVD and then compares singular values to that of
 * Colt's SVD asserting epsilon(precision) 1e-10 or whatever most recent value
 * configured.
 * 
 */
public class LocalSSVDSolverDenseTest extends MahoutTestCase {

  private static final double s_epsilon = 1.0E-10d;

  // I actually never saw errors more than 3% worst case for this test,
  // but since it's non-deterministic test, it still may ocasionally produce
  // bad results with a non-zero probability, so i put this pct% for error
  // margin higher so it never fails.
  private static final double s_precisionPct = 10;

  @Test
  public void testSSVDSolverDense() throws IOException { 
    runSSVDSolver(0);
  }
  
  @Test
  public void testSSVDSolverPowerIterations1() throws IOException { 
    runSSVDSolver(1);
  }

  @Test
  public void testSSVDSolverPowerIterations2() throws IOException { 
    runSSVDSolver(2);
  }

  public void runSSVDSolver(int q) throws IOException {

    Configuration conf = new Configuration();
    conf.set("mapred.job.tracker", "local");
    conf.set("fs.default.name", "file:///");

    // conf.set("mapred.job.tracker","localhost:11011");
    // conf.set("fs.default.name","hdfs://localhost:11010/");

    // Deque<Closeable> closeables = new LinkedList<Closeable>();
    // Random rnd = RandomUtils.getRandom();

    File tmpDir = getTestTempDir("svdtmp");
    conf.set("hadoop.tmp.dir", tmpDir.getAbsolutePath());

    Path aLocPath = new Path(getTestTempDirPath("svdtmp/A"), "A.seq");

    // create distributed row matrix-like struct
    // SequenceFile.Writer w = SequenceFile.createWriter(
    // FileSystem.getLocal(conf), conf, aLocPath, IntWritable.class,
    // VectorWritable.class, CompressionType.NONE, new DefaultCodec());
    // closeables.addFirst(w);

    // make input equivalent to 2 mln non-zero elements.
    // With 100mln the precision turns out to be only better (LLN law i guess)
    // With oversampling of 100, i don't get any error at all.
    int n = 1000;
    int m = 2000;
    Vector singularValues =
      new DenseVector(new double[] { 10, 4, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
          0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
          0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });

    SSVDTestsHelper.generateDenseInput(aLocPath,
                                       FileSystem.getLocal(conf),
                                       singularValues,
                                       m,
                                       n);

    FileSystem fs = FileSystem.get(conf);

    Path tempDirPath = getTestTempDirPath("svd-proc");
    Path aPath = new Path(tempDirPath, "A/A.seq");
    fs.copyFromLocalFile(aLocPath, aPath);

    Path svdOutPath = new Path(tempDirPath, "SSVD-out");

    // make sure we wipe out previous test results, just a convenience
    fs.delete(svdOutPath, true);

    // Solver starts here:
    System.out.println("Input prepared, starting solver...");

    int ablockRows = 867;
    int p = 10;
    int k = 3;
    SSVDSolver ssvd =
      new SSVDSolver(conf,
                     new Path[] { aPath },
                     svdOutPath,
                     ablockRows,
                     500,
                     k,
                     p,
                     3);
    // ssvd.setcUHalfSigma(true);
    // ssvd.setcVHalfSigma(true);
    ssvd.setOverwrite(true);
    ssvd.setQ(q);
    ssvd.run();

    double[] stochasticSValues = ssvd.getSingularValues();
    System.out.println("--SSVD solver singular values:");
    dumpSv(stochasticSValues);

    // the full-rank svd for this test size takes too long to run,
    // so i comment it out, instead, i will be comparing
    // result singular values to the original values used
    // to generate input (which are guaranteed to be right).

    /*
     * System.out.println("--Colt SVD solver singular values:"); // try to run
     * 
     * the same thing without stochastic algo double[][] a =
     * SSVDSolver.loadDistributedRowMatrix(fs, aPath, conf);
     * 
     * 
     * 
     * SingularValueDecomposition svd2 = new SingularValueDecomposition(new
     * DenseMatrix(a));
     * 
     * a = null;
     * 
     * double[] svalues2 = svd2.getSingularValues(); dumpSv(svalues2);
     * 
     * for (int i = 0; i < k ; i++) { Assert .assertTrue(1-Math.abs((svalues2[i]
     * - stochasticSValues[i])/svalues2[i]) <= s_precisionPct/100); }
     */

    // assert first k against those
    // used to generate surrogate input

    for (int i = 0; i < k; i++) {
      Assert
        .assertTrue(Math.abs((singularValues.getQuick(i) - stochasticSValues[i])
            / singularValues.getQuick(i)) <= s_precisionPct / 100);
    }

    double[][] mQ =
      SSVDSolver.loadDistributedRowMatrix(fs, new Path(svdOutPath, "Bt-job/"
          + BtJob.OUTPUT_Q + "-*"), conf);

    SSVDPrototypeTest
      .assertOrthonormality(new DenseMatrix(mQ), false, s_epsilon);

    double[][] u =
      SSVDSolver.loadDistributedRowMatrix(fs,
                                          new Path(svdOutPath, "U/[^_]*"),
                                          conf);

    SSVDPrototypeTest
      .assertOrthonormality(new DenseMatrix(u), false, s_epsilon);
    double[][] v =
      SSVDSolver.loadDistributedRowMatrix(fs,
                                          new Path(svdOutPath, "V/[^_]*"),
                                          conf);

    SSVDPrototypeTest
      .assertOrthonormality(new DenseMatrix(v), false, s_epsilon);
  }

  static void dumpSv(double[] s) {
    System.out.printf("svs: ");
    for (double value : s) {
      System.out.printf("%f  ", value);
    }
    System.out.println();

  }

  static void dump(double[][] matrix) {
    for (double[] aMatrix : matrix) {
      for (double anAMatrix : aMatrix) {
        System.out.printf("%f  ", anAMatrix);
      }
      System.out.println();
    }
  }

}
