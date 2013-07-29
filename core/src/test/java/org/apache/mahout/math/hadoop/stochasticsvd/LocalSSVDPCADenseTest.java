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
import java.io.IOException;
import java.util.Deque;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import com.google.common.io.Closeables;

public class LocalSSVDPCADenseTest extends MahoutTestCase {

  private static final double s_epsilon = 1.0E-10d;

  @Test
  public void runPCATest1() throws IOException {
    runSSVDSolver(1);
  }

  public void runSSVDSolver(int q) throws IOException {

    Configuration conf = new Configuration();
    conf.set("mapred.job.tracker", "local");
    conf.set("fs.default.name", "file:///");

    // conf.set("mapred.job.tracker","localhost:11011");
    // conf.set("fs.default.name","hdfs://localhost:11010/");

    Deque<Closeable> closeables = Lists.newLinkedList();
    Random rnd = RandomUtils.getRandom();

    File tmpDir = getTestTempDir("svdtmp");
    conf.set("hadoop.tmp.dir", tmpDir.getAbsolutePath());

    Path aLocPath = new Path(getTestTempDirPath("svdtmp/A"), "A.seq");

    // create distributed row matrix-like struct
    SequenceFile.Writer w =
      SequenceFile.createWriter(FileSystem.getLocal(conf),
                                conf,
                                aLocPath,
                                IntWritable.class,
                                VectorWritable.class,
                                CompressionType.BLOCK,
                                new DefaultCodec());
    closeables.addFirst(w);

    int n = 100;
    int m = 2000;
    double percent = 5;

    VectorWritable vw = new VectorWritable();
    IntWritable roww = new IntWritable();

    Vector xi = new DenseVector(n);

    double muAmplitude = 50.0;
    for (int i = 0; i < m; i++) {
      Vector dv = new SequentialAccessSparseVector(n);
      for (int j = 0; j < n * percent / 100; j++) {
        dv.setQuick(rnd.nextInt(n), muAmplitude * (rnd.nextDouble() - 0.25));
      }
      roww.set(i);
      vw.set(dv);
      w.append(roww, vw);
      xi.assign(dv, Functions.PLUS);
    }
    closeables.remove(w);
    Closeables.close(w, false);

    // TODO fix test so that 1.0/m works as intended!
    xi.assign(Functions.mult(1 / m));

    FileSystem fs = FileSystem.get(conf);

    Path tempDirPath = getTestTempDirPath("svd-proc");
    Path aPath = new Path(tempDirPath, "A/A.seq");
    fs.copyFromLocalFile(aLocPath, aPath);
    Path xiPath = new Path(tempDirPath, "xi/xi.seq");
    SSVDHelper.saveVector(xi, xiPath, conf);

    Path svdOutPath = new Path(tempDirPath, "SSVD-out");

    // make sure we wipe out previous test results, just a convenience
    fs.delete(svdOutPath, true);

    // Solver starts here:
    System.out.println("Input prepared, starting solver...");

    int ablockRows = 867;
    int p = 60;
    int k = 40;
    SSVDSolver ssvd =
      new SSVDSolver(conf,
                     new Path[] { aPath },
                     svdOutPath,
                     ablockRows,
                     k,
                     p,
                     3);
    ssvd.setOuterBlockHeight(500);
    ssvd.setAbtBlockHeight(251);
    ssvd.setPcaMeanPath(xiPath);

    /*
     * removing V,U jobs from this test to reduce running time. i will keep them
     * put in the dense test though.
     */
    ssvd.setComputeU(false);
    ssvd.setComputeV(false);

    ssvd.setOverwrite(true);
    ssvd.setQ(q);
    ssvd.setBroadcast(true);
    ssvd.run();

    Vector stochasticSValues = ssvd.getSingularValues();
    System.out.println("--SSVD solver singular values:");
    LocalSSVDSolverSparseSequentialTest.dumpSv(stochasticSValues);
    System.out.println("--Colt SVD solver singular values:");

    // try to run the same thing without stochastic algo
    double[][] a = SSVDHelper.loadDistributedRowMatrix(fs, aPath, conf);

    // subtract pseudo pca mean
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        a[i][j] -= xi.getQuick(j);
      }
    }

    SingularValueDecomposition svd2 =
      new SingularValueDecomposition(new DenseMatrix(a));

    Vector svalues2 = new DenseVector(svd2.getSingularValues());
    LocalSSVDSolverSparseSequentialTest.dumpSv(svalues2);

    for (int i = 0; i < k + p; i++) {
      assertTrue(Math.abs(svalues2.getQuick(i) - stochasticSValues.getQuick(i)) <= s_epsilon);
    }

    double[][] mQ =
      SSVDHelper.loadDistributedRowMatrix(fs, new Path(svdOutPath, "Bt-job/"
          + BtJob.OUTPUT_Q + "-*"), conf);

    SSVDCommonTest.assertOrthonormality(new DenseMatrix(mQ),
                                           false,
                                           s_epsilon);

    IOUtils.close(closeables);
  }

}
