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
package org.apache.mahout.math.ssvd;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * wraps appropriate eigen solver for BBt matrix. Can be either colt or apache
 * commons math.
 * <P>
 * 
 * At the moment it is apache commons math which is only in mahout-math
 * dependencies.
 * <P>
 * 
 * I will be happy to switch this to Colt eigensolver if it is proven reliable
 * (i experience internal errors and unsorted singular values at some point).
 * 
 * But for now commons-math seems to be more reliable.
 * 
 * 
 */
public class EigenSolverWrapper {

  private final double[] eigenvalues;
  private final double[][] uHat;

  public EigenSolverWrapper(double[][] bbt) {
    int dim = bbt.length;
    EigenDecomposition evd2 = new EigenDecomposition(new Array2DRowRealMatrix(bbt));
    eigenvalues = evd2.getRealEigenvalues();
    RealMatrix uHatrm = evd2.getV();
    uHat = new double[dim][];
    for (int i = 0; i < dim; i++) {
      uHat[i] = uHatrm.getRow(i);
    }
  }

  public double[][] getUHat() {
    return uHat;
  }

  public double[] getEigenValues() {
    return eigenvalues;
  }

}
