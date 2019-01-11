/*
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

package org.apache.mahout.math.solver;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

public final class LSMRTest extends MahoutTestCase {
  @Test
  public void basics() {
    Matrix m = hilbert(5);

    // make sure it is the hilbert matrix we know and love
    assertEquals(1, m.get(0, 0), 0);
    assertEquals(0.5, m.get(0, 1), 0);
    assertEquals(1 / 6.0, m.get(2, 3), 1.0e-9);

    Vector x = new DenseVector(new double[]{5, -120, 630, -1120, 630});

    Vector b = new DenseVector(5);
    b.assign(1);

    assertEquals(0, m.times(x).minus(b).norm(2), 1.0e-9);

    LSMR r = new LSMR();
    Vector x1 = r.solve(m, b);

    // the ideal solution is  [5  -120   630 -1120   630] but the 5x5 hilbert matrix
    // has a condition number of almost 500,000 and the normal equation condition
    // number is that squared.  This means that we don't get the exact answer with
    // a fast iterative solution.
    // Thus, we have to check the residuals rather than testing that the answer matched
    // the ideal.
    assertEquals(0, m.times(x1).minus(b).norm(2), 1.0e-2);
    assertEquals(0, m.transpose().times(m).times(x1).minus(m.transpose().times(b)).norm(2), 1.0e-7);

    // and we need to check that the error estimates are pretty good.
    assertEquals(m.times(x1).minus(b).norm(2), r.getResidualNorm(), 1.0e-5);
    assertEquals(m.transpose().times(m).times(x1).minus(m.transpose().times(b)).norm(2), r.getNormalEquationResidual(), 1.0e-9);
  }
  
  @Test
  public void random() {
    Matrix m = new DenseMatrix(200, 30).assign(Functions.random());

    Vector b = new DenseVector(200).assign(1);

    LSMR r = new LSMR();
    Vector x1 = r.solve(m, b);

//    assertEquals(0, m.times(x1).minus(b).norm(2), 1.0e-2);
    double norm = new SingularValueDecomposition(m).getS().viewDiagonal().norm(2);
    double actual = m.transpose().times(m).times(x1).minus(m.transpose().times(b)).norm(2);
    System.out.printf("%.4f\n", actual / norm * 1.0e6);
    assertEquals(0, actual, norm * 1.0e-5);

    // and we need to check that the error estimates are pretty good.
    assertEquals(m.times(x1).minus(b).norm(2), r.getResidualNorm(), 1.0e-5);
    assertEquals(actual, r.getNormalEquationResidual(), 1.0e-9);
  }

  private static Matrix hilbert(int n) {
    Matrix r = new DenseMatrix(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        r.set(i, j, 1.0 / (i + j + 1));
      }
    }
    return r;
  }

  /*
  private Matrix overDetermined(int n) {
    Random rand = RandomUtils.getRandom();
    Matrix r = new DenseMatrix(2 * n, n);
    for (int i = 0; i < 2 * n; i++) {
      for (int j = 0; j < n; j++) {
        r.set(i, j, rand.nextGaussian());
      }
    }
    return r;
  }
   */
}
