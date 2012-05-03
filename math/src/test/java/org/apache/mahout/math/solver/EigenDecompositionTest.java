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
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class EigenDecompositionTest {
  @Test
  public void testDeficientRank() {
    Matrix a = new DenseMatrix(10, 3).assign(new DoubleFunction() {
      Random gen = RandomUtils.getRandom();

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

    assertEquals(0, eig.getImagEigenvalues().norm(1), 1e-10);
    assertEquals(3, eig.getRealEigenvalues().norm(0), 1e-10);
  }

  @Test
  public void testEigen() {
    double[] evals =
      {0., 1., 0., 0.,
        1., 0., 2.e-7, 0.,
        0., -2.e-7, 0., 1.,
        0., 0., 1., 0.};
    int i = 0;
    Matrix a = new DenseMatrix(4, 4);
    for (MatrixSlice row : a) {
      for (Vector.Element element : row.vector()) {
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
    double[] columnwise = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    int i = 0;
    for (MatrixSlice row : A) {
      for (Vector.Element element : row.vector()) {
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

  private void check(String msg, Matrix a, Matrix b) {
    assertEquals(msg, 0, a.minus(b).aggregate(Functions.PLUS, Functions.ABS), 1e-10);
  }

}
