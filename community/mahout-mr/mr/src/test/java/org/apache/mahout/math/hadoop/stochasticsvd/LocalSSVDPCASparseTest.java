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

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Test;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.Deque;
import java.util.Iterator;
import java.util.Random;

public class LocalSSVDPCASparseTest extends MahoutTestCase {

  private static final double s_epsilon = 1.0E-10d;

  @Test
  public void testOmegaTRightMultiply() {
    final Random rnd = RandomUtils.getRandom();
    final long seed = rnd.nextLong();
    final int n = 2000;

    final int kp = 100;

    final Omega omega = new Omega(seed, kp);
    final Matrix materializedOmega = new DenseMatrix(n, kp);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < kp; j++)
        materializedOmega.setQuick(i, j, omega.getQuick(i, j));
    Vector xi = new DenseVector(n);
    xi.assign(new DoubleFunction() {
      @Override
      public double apply(double x) {
        return rnd.nextDouble() * 100;
      }
    });

    Vector s_o = omega.mutlithreadedTRightMultiply(xi);

    Matrix xiVector = new DenseMatrix(n, 1);
    xiVector.assignColumn(0, xi);

    Vector s_o_control = materializedOmega.transpose().times(xiVector).viewColumn(0);

    assertEquals(0, s_o.minus(s_o_control).aggregate(Functions.PLUS, Functions.ABS), 1e-10);

    System.out.printf("s_omega=\n%s\n", s_o);
    System.out.printf("s_omega_control=\n%s\n", s_o_control);
  }

  @Test
  public void runPCATest1() throws IOException {
    runSSVDSolver(1);
  }

//  @Test
  public void runPCATest0() throws IOException {
    runSSVDSolver(0);
  }


  public void runSSVDSolver(int q) throws IOException {

    Configuration conf = new Configuration();
    conf.set("mapred.job.tracker", "local");
    conf.set("fs.default.name", "file:///");

    // conf.set("mapred.job.tracker","localhost:11011");
    // conf.set("fs.default.name","hdfs://localhost:11010/");

    Deque<Closeable> closeables = Lists.newLinkedList();
    try {
      Random rnd = RandomUtils.getRandom();

      File tmpDir = getTestTempDir("svdtmp");
      conf.set("hadoop.tmp.dir", tmpDir.getAbsolutePath());

      Path aLocPath = new Path(getTestTempDirPath("svdtmp/A"), "A.seq");

      // create distributed row matrix-like struct
      SequenceFile.Writer w =
        SequenceFile.createWriter(FileSystem.getLocal(conf),
                                  conf,
                                  aLocPath,
                                  Text.class,
                                  VectorWritable.class,
                                  CompressionType.BLOCK,
                                  new DefaultCodec());
      closeables.addFirst(w);

      int n = 100;
      int m = 2000;
      double percent = 5;

      VectorWritable vw = new VectorWritable();
      Text rkey = new Text();

      Vector xi = new DenseVector(n);

      double muAmplitude = 50.0;
      for (int i = 0; i < m; i++) {
        Vector dv = new SequentialAccessSparseVector(n);
        String rowname = "row-"+i;
        NamedVector namedRow = new NamedVector(dv, rowname);
        for (int j = 0; j < n * percent / 100; j++) {
          dv.setQuick(rnd.nextInt(n), muAmplitude * (rnd.nextDouble() - 0.25));
        }
        rkey.set("row-i"+i);
        vw.set(namedRow);
        w.append(rkey, vw);
        xi.assign(dv, Functions.PLUS);
      }
      closeables.remove(w);
      Closeables.close(w, false);

      xi.assign(Functions.mult(1.0 / m));

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
                       new Path[]{aPath},
                       svdOutPath,
                       ablockRows,
                       k,
                       p,
                       3);
      ssvd.setOuterBlockHeight(500);
      ssvd.setAbtBlockHeight(251);
      ssvd.setPcaMeanPath(xiPath);

    /*
     * Removing V,U jobs from this test to reduce running time. i will keep them
     * put in the dense test though.
     *
     * For PCA test, we also want to request U*Sigma output and check it for named
     * vector propagation.
     */
      ssvd.setComputeU(false);
      ssvd.setComputeV(false);
      ssvd.setcUSigma(true);

      ssvd.setOverwrite(true);
      ssvd.setQ(q);
      ssvd.setBroadcast(true);
      ssvd.run();

      Vector stochasticSValues = ssvd.getSingularValues();

      // try to run the same thing without stochastic algo
      Matrix a = SSVDHelper.drmLoadAsDense(fs, aPath, conf);

      verifyInternals(svdOutPath, a, new Omega(ssvd.getOmegaSeed(), k + p), k + p, q);

      // subtract pseudo pca mean
      for (int i = 0; i < m; i++) {
        a.viewRow(i).assign(xi, Functions.MINUS);
      }

      SingularValueDecomposition svd2 =
        new SingularValueDecomposition(a);

      Vector svalues2 = new DenseVector(svd2.getSingularValues());

      System.out.println("--SSVD solver singular values:");
      LocalSSVDSolverSparseSequentialTest.dumpSv(stochasticSValues);
      System.out.println("--SVD solver singular values:");
      LocalSSVDSolverSparseSequentialTest.dumpSv(svalues2);

      for (int i = 0; i < k + p; i++) {
        assertTrue(Math.abs(svalues2.getQuick(i) - stochasticSValues.getQuick(i)) <= s_epsilon);
      }

      DenseMatrix mQ =
        SSVDHelper.drmLoadAsDense(fs, new Path(svdOutPath, "Bt-job/"
          + BtJob.OUTPUT_Q + "-*"), conf);

      SSVDCommonTest.assertOrthonormality(mQ,
                                          false,
                                          s_epsilon);

      // assert name propagation
      for (Iterator<Pair<Writable, Vector>> iter = SSVDHelper.drmIterator(fs,
                                                                          new Path(ssvd.getuSigmaPath()+"/*"),
                                                                          conf,
                                                                          closeables); iter.hasNext(); ) {
        Pair<Writable, Vector> pair = iter.next();
        Writable key = pair.getFirst();
        Vector v = pair.getSecond();

        assertTrue(v instanceof NamedVector);
        assertTrue(key instanceof Text);
      }

    } finally {
      IOUtils.close(closeables);
    }
  }

  private void verifyInternals(Path tempDir, Matrix a, Omega omega, int kp, int q) {
    int m = a.numRows();
    int n = a.numCols();

    Vector xi = a.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum() / v.size();
      }
    });

    // materialize omega
    Matrix momega = new DenseMatrix(n, kp);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < kp; j++)
        momega.setQuick(i, j, omega.getQuick(i, j));

    Vector s_o = omega.mutlithreadedTRightMultiply(xi);

    System.out.printf("s_omega=\n%s\n", s_o);

    Matrix y = a.times(momega);
    for (int i = 0; i < n; i++) y.viewRow(i).assign(s_o, Functions.MINUS);

    QRDecomposition qr = new QRDecomposition(y);
    Matrix qm = qr.getQ();

    Vector s_q = qm.aggregateColumns(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.zSum();
      }
    });

    System.out.printf("s_q=\n%s\n", s_q);

    Matrix b = qm.transpose().times(a);

    Vector s_b = b.times(xi);

    System.out.printf("s_b=\n%s\n", s_b);


  }

}
