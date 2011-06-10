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

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.CopyConstructorIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.ssvd.EigenSolverWrapper;

/**
 * SSVD protoptype: non-MR concept verification for Givens QR & SSVD basic
 * algorithms
 */
public class SSVDPrototype {

  private final Omega omega;
  private final int kp; // k+p
  private GivensThinSolver qSolver;
  private final double[] yRow;
  private int cnt;
  private int blckCnt;
  private final int r;
  private final List<UpperTriangular> rBlocks = Lists.newArrayList();
  private final List<double[][]> qtBlocks = Lists.newArrayList();
  private final List<double[]> yLookahead;

  public SSVDPrototype(long seed, int kp, int r) {
    this.kp = kp;
    omega = new Omega(seed, kp / 2, kp - kp / 2);
    yRow = new double[kp];
    // m_yRowV = new DenseVector(m_yRow,true);
    this.r = r;
    yLookahead = Lists.newArrayList();
  }

  void firstPass(Vector aRow) {

    omega.computeYRow(aRow, yRow);

    yLookahead.add(yRow.clone()); // bad for GC but it's just a prototype,
                                      // hey. in real thing we'll rotate usage
                                      // of y buff

    while (yLookahead.size() > kp) {

      if (qSolver == null) {
        qSolver = new GivensThinSolver(r, kp);
      }

      qSolver.appendRow(yLookahead.remove(0));
      if (qSolver.isFull()) {
        UpperTriangular r = qSolver.getRTilde();
        double[][] qt = qSolver.getThinQtTilde();
        qSolver = null;
        qtBlocks.add(qt);
        rBlocks.add(r);
      }

    }
    cnt++;
  }

  void finishFirstPass() {

    if (qSolver == null && yLookahead.isEmpty()) {
      return;
    }
    if (qSolver == null) {
      qSolver = new GivensThinSolver(yLookahead.size(), kp);
    }
    // grow q solver up if necessary

    qSolver.adjust(qSolver.getCnt() + yLookahead.size());
    while (!yLookahead.isEmpty()) {

      qSolver.appendRow(yLookahead.remove(0));
      if (qSolver.isFull()) {
        UpperTriangular r = qSolver.getRTilde();
        double[][] qt = qSolver.getThinQtTilde();
        qSolver = null;
        qtBlocks.add(qt);
        rBlocks.add(r);
      }

    }

    // simulate reducers -- produce qHats
    for (int i = 0; i < rBlocks.size(); i++) {
      qtBlocks.set(i,
                   GivensThinSolver.computeQtHat(qtBlocks.get(i), i,
                                                 new CopyConstructorIterator<UpperTriangular>(rBlocks.listIterator())));
    }
    cnt = 0;
    blckCnt = 0;
  }

  void secondPass(Vector aRow, PartialRowEmitter btEmitter)
    throws IOException {
    int n = aRow.size();
    double[][] qtHat = qtBlocks.get(blckCnt);

    int r = qtHat[0].length;
    int qRowBlckIndex = r - cnt - 1; // <-- reverse order since we fed A in
                                       // reverse
    double[] qRow = new double[kp];
    for (int i = 0; i < kp; i++) {
      qRow[i] = qtHat[i][qRowBlckIndex];
    }
    Vector qRowV = new DenseVector(qRow, true);

    if (++cnt == r) {
      blckCnt++;
      cnt = 0;
    }

    for (int i = 0; i < n; i++) {
      btEmitter.emitRow(i, qRowV.times(aRow.getQuick(i)));
    }

  }


  public static void testThinQr(int dims, int kp) throws IOException {

    DenseMatrix mx = new DenseMatrix(dims << 2, dims);
    // mx.assign(new UnaryFunction() {
    //
    // Random m_rnd = new Random(rndSeed);
    //
    // @Override
    // public double apply(double arg0) {
    // return m_rnd.nextDouble()*1000;
    // }
    // });

    Random rnd = RandomUtils.getRandom();
    for (int i = 0; i < mx.rowSize(); i++) {
      for (int j = 0; j < mx.columnSize(); j++) {
        mx.set(i, j, rnd.nextDouble() * 1000);
      }
    }

    mx.setQuick(0, 0, 1);
    mx.setQuick(0, 1, 2);
    mx.setQuick(0, 2, 3);
    mx.setQuick(1, 0, 4);
    mx.setQuick(1, 1, 5);
    mx.setQuick(1, 2, 6);
    mx.setQuick(2, 0, 7);
    mx.setQuick(2, 1, 8);
    mx.setQuick(2, 2, 9);

    SingularValueDecomposition svd2 = new SingularValueDecomposition(mx);
    double[] svaluesControl = svd2.getSingularValues();

    for (int i = 0; i < kp; i++) {
      System.out.printf("%.3e ", svaluesControl[i]);
    }
    System.out.println();

    int m = mx.rowSize(); /* ,n=mx.columnSize(); */

    long seed = RandomUtils.getRandom().nextLong();

    final Map<Integer, Vector> btRows = Maps.newHashMap();

    PartialRowEmitter btEmitter = new PartialRowEmitter() {
      @Override
      public void emitRow(int rowNum, Vector row) {
        Vector btRow = btRows.get(rowNum);
        if (btRow != null) {
          row.addTo(btRow);
        }
        btRows.put(rowNum, btRow == null ? new DenseVector(row) : btRow);
      }
    };

    SSVDPrototype mapperSimulation = new SSVDPrototype(seed, kp, 3000);
    for (int i = 0; i < m; i++) {
      mapperSimulation.firstPass(mx.getRow(i));
    }

    mapperSimulation.finishFirstPass();

    for (int i = 0; i < m; i++) {
      mapperSimulation.secondPass(mx.getRow(i), btEmitter);
    }

    // LocalSSVDTest.assertOrthonormality(mapperSimulation.m_qt.transpose(),
    // false,1e-10);

    // reconstruct bbt
    final Map<Integer, Vector> bbt = Maps.newHashMap();
    PartialRowEmitter bbtEmitter = new PartialRowEmitter() {

      @Override
      public void emitRow(int rowNum, Vector row) {
        Vector bbtRow = bbt.get(rowNum);
        if (bbtRow != null) {
          row.addTo(bbtRow);
        }
        bbt.put(rowNum, bbtRow == null ? new DenseVector(row) : bbtRow);
      }
    };

    for (Map.Entry<Integer, Vector> btRowEntry : btRows.entrySet()) {
      Vector btRow = btRowEntry.getValue();
      assert btRow.size() == kp;
      for (int i = 0; i < kp; i++) {
        bbtEmitter.emitRow(i, btRow.times(btRow.getQuick(i)));
      }
    }

    double[][] bbtValues = new double[kp][];
    for (int i = 0; i < kp; i++) {
      bbtValues[i] = new double[kp];
      Vector bbtRow = bbt.get(i);
      for (int j = 0; j < kp; j++) {
        bbtValues[i][j] = bbtRow.getQuick(j);
      }
    }

    EigenSolverWrapper eigenWrapper = new EigenSolverWrapper(bbtValues);
    double[] eigenva2 = eigenWrapper.getEigenValues();
    double[] svalues = new double[kp];
    for (int i = 0; i < kp; i++) {
      svalues[i] = Math.sqrt(eigenva2[i]); // sqrt?
    }

    for (int i = 0; i < kp; i++) {
      System.out.printf("%.3e ", svalues[i]);
    }
    System.out.println();

  }

  public static void testBlockQrWithSSVD(int dims, int kp, int r, long rndSeed) throws IOException {

    DenseMatrix mx = new DenseMatrix(dims << 2, dims);
    // mx.assign(new UnaryFunction() {
    //
    // Random m_rnd = new Random(rndSeed);
    //
    // @Override
    // public double apply(double arg0) {
    // return (m_rnd.nextDouble()-0.5)*1000;
    // }
    // });

    Random rnd = RandomUtils.getRandom();
    for (int i = 0; i < mx.rowSize(); i++) {
      for (int j = 0; j < mx.columnSize(); j++) {
        mx.set(i, j, (rnd.nextDouble() - 0.5) * 1000);
      }
    }
    mx.setQuick(0, 0, 1);
    mx.setQuick(0, 1, 2);
    mx.setQuick(0, 2, 3);
    mx.setQuick(1, 0, 4);
    mx.setQuick(1, 1, 5);
    mx.setQuick(1, 2, 6);
    mx.setQuick(2, 0, 7);
    mx.setQuick(2, 1, 8);
    mx.setQuick(2, 2, 9);

    SingularValueDecomposition svd2 = new SingularValueDecomposition(mx);
    double[] svaluesControl = svd2.getSingularValues();

    for (int i = 0; i < kp; i++) {
      System.out.printf("%e ", svaluesControl[i]);
    }
    System.out.println();

    int m = mx.rowSize(); /* ,n=mx.columnSize(); */

    final Map<Integer, Vector> btRows = Maps.newHashMap();

    PartialRowEmitter btEmitter = new PartialRowEmitter() {
      @Override
      public void emitRow(int rowNum, Vector row) {
        Vector btRow = btRows.get(rowNum);
        if (btRow != null) {
          row.addTo(btRow);
        }
        btRows.put(rowNum, btRow == null ? new DenseVector(row) : btRow);
      }
    };

    SSVDPrototype mapperSimulation = new SSVDPrototype(rndSeed, kp, r);
    for (int i = 0; i < m; i++) {
      mapperSimulation.firstPass(mx.getRow(i));
    }

    mapperSimulation.finishFirstPass();

    for (int i = 0; i < m; i++) {
      mapperSimulation.secondPass(mx.getRow(i), btEmitter);
    }

    // LocalSSVDTest.assertOrthonormality(mapperSimulation.m_qt.transpose(),
    // false,1e-10);

    // reconstruct bbt
    final Map<Integer, Vector> bbt = Maps.newHashMap();
    PartialRowEmitter bbtEmitter = new PartialRowEmitter() {

      @Override
      public void emitRow(int rowNum, Vector row) {
        Vector bbtRow = bbt.get(rowNum);
        if (bbtRow != null) {
          row.addTo(bbtRow);
        }
        bbt.put(rowNum, bbtRow == null ? new DenseVector(row) : bbtRow);
      }
    };

    for (Map.Entry<Integer, Vector> btRowEntry : btRows.entrySet()) {
      Vector btRow = btRowEntry.getValue();
      assert btRow.size() == kp;
      for (int i = 0; i < kp; i++) {
        bbtEmitter.emitRow(i, btRow.times(btRow.getQuick(i)));
      }
    }

    double[][] bbtValues = new double[kp][];
    for (int i = 0; i < kp; i++) {
      bbtValues[i] = new double[kp];
      Vector bbtRow = bbt.get(i);
      for (int j = 0; j < kp; j++) {
        bbtValues[i][j] = bbtRow.getQuick(j);
      }
    }

    EigenSolverWrapper eigenWrapper = new EigenSolverWrapper(bbtValues);

    double[] eigenva2 = eigenWrapper.getEigenValues();
    double[] svalues = new double[kp];
    for (int i = 0; i < kp; i++) {
      svalues[i] = Math.sqrt(eigenva2[i]); // sqrt?
    }

    for (int i = 0; i < kp; i++) {
      System.out.printf("%e ", svalues[i]);
    }
    System.out.println();
  }

  public static void main(String[] args) throws Exception {
    // testThinQr();
    long seed = RandomUtils.getRandom().nextLong();
    testBlockQrWithSSVD(200, 200, 800, seed);
    testBlockQrWithSSVD(200, 20, 800, seed);
    testBlockQrWithSSVD(200, 20, 850, seed); // test trimming
    testBlockQrWithSSVD(200, 20, 90, seed);
    testBlockQrWithSSVD(200, 20, 99, seed);
  }

}
