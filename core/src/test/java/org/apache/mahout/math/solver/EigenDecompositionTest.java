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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class EigenDecompositionTest {
  @Test
  public void testDegenerateMatrix() {
    double[][] m = {
      new double[]{0.641284, 0.767303, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000},
      new double[]{0.767303, 3.050159, 2.561342, 0.000000, 0.000000, 0.000000, 0.000000},
      new double[]{0.000000, 2.561342, 5.000609, 0.810507, 0.000000, 0.000000, 0.000000},
      new double[]{0.000000, 0.000000, 0.810507, 0.550477, 0.142853, 0.000000, 0.000000},
      new double[]{0.000000, 0.000000, 0.000000, 0.142853, 0.254566, 0.000000, 0.000000},
      new double[]{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.256073, 0.000000},
      new double[]{0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000}};
    Matrix x = new DenseMatrix(m);
    EigenDecomposition eig = new EigenDecomposition(x, true);
    Matrix d = eig.getD();
    Matrix v = eig.getV();
    check("EigenvalueDecomposition (evil)...", x.times(v), v.times(d));
  }

  @Test
  public void testDeficientRank() {
    Matrix a = new DenseMatrix(10, 3).assign(new DoubleFunction() {
      private final Random gen = RandomUtils.getRandom();
      @Override
      public double apply(double arg1) {
        return gen.nextGaussian();
      }
    });

    a = a.transpose().times(a);

    EigenDecomposition eig = new EigenDecomposition(a);
    Matrix d = eig.getD();
    Matrix v = eig.getV();
    check("EigenvalueDecomposition (rank deficient)...", a.times(v), v.times(d));

    Assert.assertEquals(0, eig.getImagEigenvalues().norm(1), 1.0e-10);
    Assert.assertEquals(3, eig.getRealEigenvalues().norm(0), 1.0e-10);
  }

  @Test
  public void testEigen() {
    double[] evals =
      {0.0, 1.0, 0.0, 0.0,
          1.0, 0.0, 2.0e-7, 0.0,
          0.0, -2.0e-7, 0.0, 1.0,
          0.0, 0.0, 1.0, 0.0};
    int i = 0;
    Matrix a = new DenseMatrix(4, 4);
    for (MatrixSlice row : a) {
      for (Vector.Element element : row.vector().all()) {
        element.set(evals[i++]);
      }
    }
    EigenDecomposition eig = new EigenDecomposition(a);
    Matrix d = eig.getD();
    Matrix v = eig.getV();
    check("EigenvalueDecomposition (nonsymmetric)...", a.times(v), v.times(d));
  }

  @Test
  public void testSequential() {
    int validld = 3;
    Matrix A = new DenseMatrix(validld, validld);
    double[] columnwise = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    int i = 0;
    for (MatrixSlice row : A) {
      for (Vector.Element element : row.vector().all()) {
        element.set(columnwise[i++]);
      }
    }

    EigenDecomposition Eig = new EigenDecomposition(A);
    Matrix D = Eig.getD();
    Matrix V = Eig.getV();
    check("EigenvalueDecomposition (nonsymmetric)...", A.times(V), V.times(D));

    A = A.transpose().times(A);
    Eig = new EigenDecomposition(A);
    D = Eig.getD();
    V = Eig.getV();
    check("EigenvalueDecomposition (symmetric)...", A.times(V), V.times(D));

  }

  private static void check(String msg, Matrix a, Matrix b) {
    Assert.assertEquals(msg, 0, a.minus(b).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
  }

}
