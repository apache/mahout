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

package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class CholeskyDecompositionTest extends MahoutTestCase {
  @Test
  public void rank1() {
    Matrix x = new DenseMatrix(3, 3);
    x.viewRow(0).assign(new double[]{1, 2, 3});
    x.viewRow(1).assign(new double[]{2, 4, 6});
    x.viewRow(2).assign(new double[]{3, 6, 9});

    CholeskyDecomposition rr = new CholeskyDecomposition(x.transpose().times(x), false);
    assertEquals(0, new DenseVector(new double[]{3.741657, 7.483315, 11.22497}).aggregate(rr.getL().transpose().viewRow(0), Functions.PLUS, new DoubleDoubleFunction() {
      @Override
      public double apply(double arg1, double arg2) {
        return Math.abs(arg1) - Math.abs(arg2);
      }
    }), 1.0e-5);

    assertEquals(0, rr.getL().viewPart(0, 3, 1, 2).aggregate(Functions.PLUS, Functions.ABS), 1.0e-9);
  }

  @Test
  public void test1() {

    final Random rand = RandomUtils.getRandom();

    Matrix z = new DenseMatrix(100, 100);
    z.assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return rand.nextDouble();
      }
    });

    Matrix A = z.times(z.transpose());

    for (boolean type = false; !type; type=true) {
      CholeskyDecomposition cd = new CholeskyDecomposition(A, type);
      Matrix L = cd.getL();
//      Assert.assertTrue("Positive definite", cd.isPositiveDefinite());

      Matrix Abar = L.times(L.transpose());

      double error = A.minus(Abar).aggregate(Functions.MAX, Functions.ABS);
      Assert.assertEquals("type = " + type, 0, error, 1.0e-10);

      // L should give us a quick and dirty LQ decomposition
      Matrix q = cd.solveLeft(z);
      Matrix id = q.times(q.transpose());
      for (int i = 0; i < id.columnSize(); i++) {
        Assert.assertEquals("type = " + type, 1, id.get(i, i), 1.0e-9);
        Assert.assertEquals("type = " + type, 1, id.viewRow(i).norm(1), 1.0e-9);
      }

      // and QR as well
      q = cd.solveRight(z.transpose());
      id = q.transpose().times(q);
      for (int i = 0; i < id.columnSize(); i++) {
        Assert.assertEquals("type = " + type, 1, id.get(i, i), 1.0e-9);
        Assert.assertEquals("type = " + type, 1, id.viewRow(i).norm(1), 1.0e-9);
      }
    }
  }

  @Test
  public void test2() {
    // Test matrix from Nicholas Higham's paper at http://eprints.ma.man.ac.uk/1199/01/covered/MIMS_ep2008_116.pdf
    double[][] values = new double[3][];
    values[0] = new double[]{1, -1, 1};
    values[1] = new double[]{-1, 1, -1};
    values[2] = new double[]{1, -1, 2};

    Matrix A = new DenseMatrix(values);

    // without pivoting
    CholeskyDecomposition cd = new CholeskyDecomposition(A, false);
    assertEquals(0, cd.getL().times(cd.getL().transpose()).minus(A).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

    // with pivoting
    cd = new CholeskyDecomposition(A);
    assertEquals(0, cd.getL().times(cd.getL().transpose()).minus(A).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
  }


  @Test
  public void testRankDeficient() {
    Matrix A = rank4Matrix();

    CholeskyDecomposition cd = new CholeskyDecomposition(A);

    PivotedMatrix Ax = new PivotedMatrix(A, cd.getPivot());
    CholeskyDecomposition cd2 = new CholeskyDecomposition(Ax, false);

    assertEquals(0, cd2.getL().times(cd2.getL().transpose()).minus(Ax).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
    assertEquals(0, cd.getL().times(cd.getL().transpose()).minus(A).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

    Assert.assertFalse(cd.isPositiveDefinite());
    Matrix L = cd.getL();
    Matrix Abar = L.times(L.transpose());
    double error = A.minus(Abar).aggregate(Functions.MAX, Functions.ABS);
    Assert.assertEquals(0, error, 1.0e-10);
  }

  private static Matrix rank4Matrix() {
    final Random rand = RandomUtils.getRandom();

    Matrix u = new DenseMatrix(10, 4);
    u.assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return rand.nextDouble();
      }
    });

    Matrix v = new DenseMatrix(10, 4);
    v.assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return rand.nextDouble();
      }
    });

    Matrix z = u.times(v.transpose());
    return z.times(z.transpose());
  }
}
