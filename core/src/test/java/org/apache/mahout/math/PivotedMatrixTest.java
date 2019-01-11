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

import org.junit.Test;

public class PivotedMatrixTest extends MatrixTest {
  @Override
  public Matrix matrixFactory(double[][] values) {

    Matrix base = new DenseMatrix(values);

    // for general tests, we just make a scrambled matrix and fill it
    // with the standard data.  Then we can test the details of the
    // row and/or column swapping separately.
    PivotedMatrix pm = new PivotedMatrix(base.like());

    pm.swap(0, 1);
    pm.swapRows(1, 2);
    pm.assign(base);
    return pm;
  }

  @Test
  public void testSwap() {
    Matrix m = new DenseMatrix(10, 10);
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        m.set(i, j, 10 * i + j);
      }
    }

    PivotedMatrix pm = new PivotedMatrix(m);

    pm.swap(3, 5);

    assertEquals(0, pm.viewDiagonal().minus(
        new DenseVector(new double[]{0, 11, 22, 55, 44, 33, 66, 77, 88, 99})).norm(1), 1.0e-10);

    pm.swap(2, 7);
    assertEquals(0, pm.viewDiagonal().minus(
        new DenseVector(new double[]{0, 11, 77, 55, 44, 33, 66, 22, 88, 99})).norm(1), 1.0e-10);

    pm.swap(5, 8);
    assertEquals(0, pm.viewColumn(4).minus(
        new DenseVector(new double[]{4.0,14.0,74.0,54.0,44.0,84.0,64.0,24.0,34.0,94.0})).norm(1), 1.0e-10);
    assertEquals(0, pm.viewDiagonal().minus(
        new DenseVector(new double[]{0, 11, 77, 55, 44, 88, 66, 22, 33, 99})).norm(1), 1.0e-10);
  }
}
