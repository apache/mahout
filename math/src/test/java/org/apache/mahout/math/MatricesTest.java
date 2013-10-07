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

package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.IntIntFunction;
import org.junit.Test;

public class MatricesTest extends MahoutTestCase {

  @Test
  public void testFunctionalView() {
    Matrix m = Matrices.functionalMatrixView(5, 6, new IntIntFunction() {
      @Override
      public double apply(int row, int col) {
        assertTrue(row < 5);
        assertTrue(col < 6);
        return row + col;
      }
    });

    // row-wise sums are 15, 15+ 6, 15 +12, 15+18, 15+24
    // so total sum is 1/2*(15+15+24)*5 =27*5 = 135
    assertEquals(135, m.aggregate(Functions.PLUS, Functions.IDENTITY), 1e-10);
  }

  @Test
  public void testTransposeView() {

    Matrix m = Matrices.gaussianView(5, 6, 1234L);
    Matrix controlM = new DenseMatrix(5, 6).assign(m);

    System.out.printf("M=\n%s\n", m);
    System.out.printf("controlM=\n%s\n", controlM);

    Matrix mtm = Matrices.transposedView(m).times(m);
    Matrix controlMtm = controlM.transpose().times(controlM);

    System.out.printf("M'M=\n%s\n", mtm);

    Matrix diff = mtm.minus(controlMtm);

    assertEquals(0, diff.aggregate(Functions.PLUS, Functions.ABS), 1e-10);

  }

  @Test
  public void testUniformView() {
    Matrix m1 = Matrices.uniformView(5, 6, 1234);
    Matrix m2 = Matrices.uniformView(5, 6, 1234);

    for (int row = 0; row < m1.numRows(); row++) {
      for (int col = 0; col < m1.numCols(); col++) {
        assertTrue(m1.getQuick(row, col) >= 0.0);
        assertTrue(m1.getQuick(row, col) < 1.0);
      }
    }

    Matrix diff = m1.minus(m2);

    assertEquals(0, diff.aggregate(Functions.PLUS, Functions.ABS), 1e-10);
  }

  @Test
  public void testSymmetricUniformView() {
    Matrix m1 = Matrices.symmetricUniformView(5, 6, 1234);
    Matrix m2 = Matrices.symmetricUniformView(5, 6, 1234);

    for (int row = 0; row < m1.numRows(); row++) {
      for (int col = 0; col < m1.numCols(); col++) {
        assertTrue(m1.getQuick(row, col) >= -1.0);
        assertTrue(m1.getQuick(row, col) < 1.0);
      }
    }

    Matrix diff = m1.minus(m2);

    assertEquals(0, diff.aggregate(Functions.PLUS, Functions.ABS), 1e-10);
  }

  @Test
  public void testGaussianView() {
    Matrix m1 = Matrices.gaussianView(5, 6, 1234);
    Matrix m2 = Matrices.gaussianView(5, 6, 1234);

    Matrix diff = m1.minus(m2);

    assertEquals(0, diff.aggregate(Functions.PLUS, Functions.ABS), 1e-10);
  }

}
