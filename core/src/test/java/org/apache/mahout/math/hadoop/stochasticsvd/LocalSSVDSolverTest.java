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

import java.io.Closeable;
import java.io.File;
import java.util.Deque;
import java.util.LinkedList;
import java.util.Random;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * 
 * Tests SSVD solver with a made-up data running hadoop 
 * solver in a local mode. It requests full-rank SSVD and 
 * then compares singular values to that of Colt's SVD 
 * asserting epsilon(precision) 1e-10 or whatever most recent 
 * value configured. 
 * 
 */
public class LocalSSVDSolverTest extends MahoutTestCase {

  private static final double s_epsilon = 1.0E-10d;

  @Test
  public void testSSVDSolver() throws Exception {

    Configuration conf = new Configuration();
    conf.set("mapred.job.tracker", "local");
    conf.set("fs.default.name", "file:///");

    // conf.set("mapred.job.tracker","localhost:11011");
    // conf.set("fs.default.name","hdfs://localhost:11010/");

    Deque<Closeable> closeables = new LinkedList<Closeable>();
    Random rnd = RandomUtils.getRandom();

    File tmpDir = getTestTempDir("svdtmp");
    conf.set("hadoop.tmp.dir", tmpDir.getAbsolutePath());

    Path aLocPath = new Path(getTestTempDirPath("svdtmp/A"), "A.seq");

    // create distributed row matrix-like struct
    SequenceFile.Writer w = SequenceFile.createWriter(
        FileSystem.getLocal(conf), conf, aLocPath, IntWritable.class,
        VectorWritable.class, CompressionType.BLOCK, new DefaultCodec());
    closeables.addFirst(w);

    int n = 100;
    double[] row = new double[n];
    DenseVector dv = new DenseVector(row, true);
    Writable vw = new VectorWritable(dv);
    IntWritable roww = new IntWritable();

    double muAmplitude = 50.0;
    int m = 1000;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        row[j] = muAmplitude * (rnd.nextDouble() - 0.5);
      }
      roww.set(i);
      w.append(roww, vw);
    }
    closeables.remove(w);
    w.close();

    FileSystem fs = FileSystem.get(conf);

    Path tempDirPath = getTestTempDirPath("svd-proc");
    Path aPath = new Path(tempDirPath, "A/A.seq");
    fs.copyFromLocalFile(aLocPath, aPath);

    Path svdOutPath = new Path(tempDirPath, "SSVD-out");

    // make sure we wipe out previous test results, just a convenience
    fs.delete(svdOutPath, true);

    int ablockRows = 251;
    int p = 60;
    int k = 40;
    SSVDSolver ssvd = new SSVDSolver(conf, new Path[] { aPath }, svdOutPath,
        ablockRows, k, p, 3);
    // ssvd.setcUHalfSigma(true);
    // ssvd.setcVHalfSigma(true);
    ssvd.setOverwrite(true);
    ssvd.run();

    double[] stochasticSValues = ssvd.getSingularValues();
    System.out.println("--SSVD solver singular values:");
    dumpSv(stochasticSValues);
    System.out.println("--Colt SVD solver singular values:");

    // try to run the same thing without stochastic algo
    double[][] a = SSVDSolver.loadDistributedRowMatrix(fs, aPath, conf);

    // SingularValueDecompositionImpl svd=new SingularValueDecompositionImpl(new
    // Array2DRowRealMatrix(a));
    SingularValueDecomposition svd2 = new SingularValueDecomposition(
        new DenseMatrix(a));

    a = null;

    double[] svalues2 = svd2.getSingularValues();
    dumpSv(svalues2);

    for (int i = 0; i < k + p; i++) {
      Assert.assertTrue(Math.abs(svalues2[i] - stochasticSValues[i]) <= s_epsilon);
    }

    double[][] q = SSVDSolver.loadDistributedRowMatrix(fs, new Path(svdOutPath,
        "Bt-job/" + BtJob.OUTPUT_Q + "-*"), conf);

    SSVDPrototypeTest.assertOrthonormality(new DenseMatrix(q), false, s_epsilon);

    double[][] u = SSVDSolver.loadDistributedRowMatrix(fs, new Path(svdOutPath,
                                                                    "U/[^_]*"), conf);

    SSVDPrototypeTest.assertOrthonormality(new DenseMatrix(u), false, s_epsilon);
    double[][] v = SSVDSolver.loadDistributedRowMatrix(fs, new Path(svdOutPath,
        "V/[^_]*"), conf);

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
