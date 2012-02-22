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

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.hadoop.stochasticsvd.qr.GivensThinSolver;
import org.junit.Test;

/**
 * Shared ssvd test code
 */
public class SSVDCommonTest extends MahoutTestCase {

  private static final double SCALE = 1000;
  private static final double SVD_EPSILON = 1.0e-10;

  @Test
  public void testGivensQR() throws Exception {
    // DenseMatrix m = new DenseMatrix(dims<<2,dims);
    Matrix m = new DenseMatrix(3, 3);
    m.assign(new DoubleFunction() {
      private final Random rnd = RandomUtils.getRandom();

      @Override
      public double apply(double arg0) {
        return rnd.nextDouble() * SCALE;
      }
    });

    m.setQuick(0, 0, 1);
    m.setQuick(0, 1, 2);
    m.setQuick(0, 2, 3);
    m.setQuick(1, 0, 4);
    m.setQuick(1, 1, 5);
    m.setQuick(1, 2, 6);
    m.setQuick(2, 0, 7);
    m.setQuick(2, 1, 8);
    m.setQuick(2, 2, 9);

    GivensThinSolver qrSolver =
      new GivensThinSolver(m.rowSize(), m.columnSize());
    qrSolver.solve(m);

    Matrix qtm = new DenseMatrix(qrSolver.getThinQtTilde());

    assertOrthonormality(qtm.transpose(), false, SVD_EPSILON);

    Matrix aClone =
      new DenseMatrix(qrSolver.getThinQtTilde()).transpose()
                                                .times(qrSolver.getRTilde());

    System.out.println("aclone : " + aClone);

  }

  public static void assertOrthonormality(Matrix mtx,
                                          boolean insufficientRank,
                                          double epsilon) {
    int n = mtx.columnSize();
    int rank = 0;
    for (int i = 0; i < n; i++) {
      Vector ei = mtx.viewColumn(i);

      double norm = ei.norm(2);

      if (Math.abs(1 - norm) < epsilon) {
        rank++;
      } else {
        assertTrue(Math.abs(norm) < epsilon);
      }

      for (int j = 0; j <= i; j++) {
        Vector e_j = mtx.viewColumn(j);
        double dot = ei.dot(e_j);
        assertTrue(Math.abs((i == j && rank > j ? 1 : 0) - dot) < epsilon);
      }
    }
    assertTrue((!insufficientRank && rank == n)
        || (insufficientRank && rank < n));

  }

}
