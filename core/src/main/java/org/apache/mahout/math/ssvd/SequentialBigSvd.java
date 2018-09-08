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

import org.apache.mahout.math.CholeskyDecomposition;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomTrinaryMatrix;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;

/**
 * Implements an in-memory version of stochastic projection based SVD.  See SequentialOutOfCoreSvd
 * for algorithm notes.
 */
public class SequentialBigSvd {
  private final Matrix y;
  private final CholeskyDecomposition cd1;
  private final CholeskyDecomposition cd2;
  private final SingularValueDecomposition svd;
  private final Matrix b;


  public SequentialBigSvd(Matrix A, int p) {
    // Y = A * \Omega
    y = A.times(new RandomTrinaryMatrix(A.columnSize(), p));

    // R'R = Y' Y
    cd1 = new CholeskyDecomposition(y.transpose().times(y));

    // B = Q" A = (Y R^{-1} )' A
    b = cd1.solveRight(y).transpose().times(A);

    // L L' = B B'
    cd2 = new CholeskyDecomposition(b.times(b.transpose()));

    // U_0 D V_0' = L
    svd = new SingularValueDecomposition(cd2.getL());
  }

  public Vector getSingularValues() {
    return new DenseVector(svd.getSingularValues());
  }

  public Matrix getU() {
    // U = (Y inv(R)) U_0
    return cd1.solveRight(y).times(svd.getU());
  }

  public Matrix getV() {
    // V = (B' inv(L')) V_0
    return cd2.solveRight(b.transpose()).times(svd.getV());
  }
}
