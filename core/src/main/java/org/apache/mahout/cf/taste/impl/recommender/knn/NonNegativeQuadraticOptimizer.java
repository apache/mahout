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

package org.apache.mahout.cf.taste.impl.recommender.knn;

import java.util.Arrays;

/**
 * Non-negative Quadratic Optimization. Based on the paper of Robert M. Bell and Yehuda Koren in ICDM '07. Thanks to Dan
 * Tillberg for the hints in the implementation.
 */
public final class NonNegativeQuadraticOptimizer implements Optimizer {

  /**
   * Non-negative Quadratic Optimization.
   *
   * @param A matrix nxn positions
   * @param b vector b, n positions
   * @return vector of n weights
   */
  @Override
  public double[] optimize(double[][] A, double[] b) {
    int k = b.length;
    double[] r = new double[k];
    double[] x = new double[k];
    Arrays.fill(x, 3.0 / (double) k);

    double rdot;
    do {

      rdot = 0.0;
      for (int n = 0; n < k; n++) {
        double sumAw = 0.0;
        double[] An = A[n];
        for (int i = 0; i < k; i++) {
          sumAw += An[i] * x[i];
        }
        // r = b - Ax; // the residual, or 'steepest gradient'
        r[n] = b[n] - sumAw;

        // find active variables - those that are pinned due to
        // nonnegativity constraints; set respective ri's to zero
        if ((x[n] < 1.0e-10) && (r[n] < 0.0)) {
          r[n] = 0.0;
        }

        // max step size numerator
        rdot += r[n] * r[n];
      }

      // max step size denominator
      double rArdotSum = 0.0;
      for (int n = 0; n < k; n++) {
        double SumAr = 0.0;
        double[] An = A[n];
        for (int i = 0; i < k; i++) {
          SumAr += An[i] * r[i];
        }
        rArdotSum += r[n] * SumAr;
      }

      // max step size
      double stepSize = rdot / rArdotSum;

      if (Double.isNaN(stepSize)) {
        stepSize = 0.001;
      }

      // adjust step size to prevent negative values
      for (int n = 0; n < k; n++) {
        if (r[n] < 0.0) {
          stepSize = Math.min(Math.abs(stepSize), Math.abs(x[n] / r[n])) * stepSize / Math.abs(stepSize);
        }
      }

      // update x values
      for (int n = 0; n < k; n++) {
        x[n] += stepSize * r[n];
        if (x[n] < 1.0e-10) {
          x[n] = 0.0;
        }
      }

      /*
      if (rdot > (20 * k) || Double.isNaN(rdot) || (iteration > 5000)) {
        //TODO: do something in case of divergence
      }
       */
    } while (rdot > 0.1);

    return x;
  }

}
