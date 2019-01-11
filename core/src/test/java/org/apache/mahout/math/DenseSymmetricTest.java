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

import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.solver.EigenDecomposition;
import org.junit.Test;

public class DenseSymmetricTest extends MahoutTestCase {
  @Test
  public void testBasics() {
    Matrix a = new DenseSymmetricMatrix(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, false);
    System.out.println(a.toString());

    assertEquals(0, a.viewDiagonal().minus(new DenseVector(new double[]{1, 5, 8, 10})).norm(1), 1.0e-10);
    assertEquals(0, a.viewPart(0, 3, 1, 3).viewDiagonal().minus(
        new DenseVector(new double[]{2, 6, 9})).norm(1), 1.0e-10);
    assertEquals(4, a.get(0, 3), 1.0e-10);
    System.out.println(a);

    Matrix m = new DenseMatrix(4, 4).assign(a);
    assertEquals(0, m.minus(a).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
    System.out.println(m);

    assertEquals(0, m.transpose().times(m).minus(a.transpose().times(a)).aggregate(
        Functions.PLUS, Functions.ABS), 1.0e-10);

    System.out.println(a.plus(a));
    assertEquals(0, m.plus(m).minus(a.plus(a)).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
  }

  @Test
  public void testEigen() {
    Matrix a = new DenseSymmetricMatrix(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, false);
    Matrix b = new DenseMatrix(a.numRows(), a.numCols());
    b.assign(a);

    assertEquals(0, a.minus(b).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

    EigenDecomposition edA = new EigenDecomposition(a);
    EigenDecomposition edB = new EigenDecomposition(b);

    System.out.println(edA.getV());

    assertEquals(0, edA.getV().minus(edB.getV()).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);
    assertEquals(0, edA.getRealEigenvalues().minus(edA.getRealEigenvalues()).aggregate(Functions.PLUS, Functions.ABS), 1.0e-10);

  }

}
