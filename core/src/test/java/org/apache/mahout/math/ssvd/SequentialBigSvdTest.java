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

package org.apache.mahout.math.ssvd;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalMatrix;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

public final class SequentialBigSvdTest extends MahoutTestCase {

  @Test
  public void testSingularValues() {
    Matrix A = lowRankMatrix();

    SequentialBigSvd s = new SequentialBigSvd(A, 8);
    SingularValueDecomposition svd = new SingularValueDecomposition(A);

    Vector reference = new DenseVector(svd.getSingularValues()).viewPart(0, 8);
    assertEquals(reference, s.getSingularValues());

    assertEquals(A, s.getU().times(new DiagonalMatrix(s.getSingularValues())).times(s.getV().transpose()));
  }

  @Test
  public void testLeftVectors() {
    Matrix A = lowRankMatrix();

    SequentialBigSvd s = new SequentialBigSvd(A, 8);
    SingularValueDecomposition svd = new SingularValueDecomposition(A);

    // can only check first few singular vectors because once the singular values
    // go to zero, the singular vectors are not uniquely determined
    Matrix u1 = svd.getU().viewPart(0, 20, 0, 4).assign(Functions.ABS);
    Matrix u2 = s.getU().viewPart(0, 20, 0, 4).assign(Functions.ABS);
    assertEquals(0, u1.minus(u2).aggregate(Functions.PLUS, Functions.ABS), 1.0e-9);
  }

  private static void assertEquals(Matrix u1, Matrix u2) {
    assertEquals(0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1.0e-10);
  }

  private static void assertEquals(Vector u1, Vector u2) {
    assertEquals(0, u1.minus(u2).aggregate(Functions.MAX, Functions.ABS), 1.0e-10);
  }

  @Test
  public void testRightVectors() {
    Matrix A = lowRankMatrix();

    SequentialBigSvd s = new SequentialBigSvd(A, 6);
    SingularValueDecomposition svd = new SingularValueDecomposition(A);

    Matrix v1 = svd.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    Matrix v2 = s.getV().viewPart(0, 20, 0, 3).assign(Functions.ABS);
    assertEquals(v1, v2);
  }

  private static Matrix lowRankMatrix() {
    Matrix u = new RandomTrinaryMatrix(1, 20, 4, false);
    Matrix d = new DiagonalMatrix(new double[]{5, 3, 1, 0.5});
    Matrix v = new RandomTrinaryMatrix(2, 23, 4, false);

    return u.times(d).times(v.transpose());
  }
}
