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
 * Non-negative Quadratic Optimization. Based on the paper of Robert M. Bell and Yehuda Koren in ICDM '07.
 * Thanks to Dan Tillberg for the hints in the implementation.
 */
public final class NonNegativeQuadraticOptimizer implements Optimizer {
  
  private static final double EPSILON = 1.0e-10;
  private static final double CONVERGENCE_LIMIT = 0.1;
  private static final int MAX_ITERATIONS = 1000;
  private static final double DEFAULT_STEP = 0.001;
  
  /**
   * Non-negative Quadratic Optimization.
   * 
   * @param matrix
   *          matrix nxn positions
   * @param b
   *          vector b, n positions
   * @return vector of n weights
   */
  @Override
  public double[] optimize(double[][] matrix, double[] b) {
    int k = b.length;
    double[] r = new double[k];
    double[] x = new double[k];
    Arrays.fill(x, 3.0 / k);
    
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
      
      double rdot = 0.0;
      for (int n = 0; n < k; n++) {
        double sumAw = 0.0;
        double[] rowAn = matrix[n];
        for (int i = 0; i < k; i++) {
          sumAw += rowAn[i] * x[i];
        }
        // r = b - Ax; // the residual, or 'steepest gradient'
        double rn = b[n] - sumAw;
        
        // find active variables - those that are pinned due to
        // nonnegativity constraints; set respective ri's to zero
        if (x[n] < EPSILON && rn < 0.0) {
          rn = 0.0;
        } else {
          // max step size numerator
          rdot += rn * rn;
        }
        r[n] = rn;
      }
      
      if (rdot <= CONVERGENCE_LIMIT) {
        break;
      }
      
      // max step size denominator
      double rArdotSum = 0.0;
      for (int n = 0; n < k; n++) {
        double sumAr = 0.0;
        double[] rowAn = matrix[n];
        for (int i = 0; i < k; i++) {
          sumAr += rowAn[i] * r[i];
        }
        rArdotSum += r[n] * sumAr;
      }
      
      // max step size
      double stepSize = rdot / rArdotSum;
      
      if (Double.isNaN(stepSize)) {
        stepSize = DEFAULT_STEP;
      }
      
      // adjust step size to prevent negative values
      for (int n = 0; n < k; n++) {
        if (r[n] < 0.0) {
          double absStepSize = stepSize < 0.0 ? -stepSize : stepSize;
          stepSize = Math.min(absStepSize, Math.abs(x[n] / r[n])) * stepSize / absStepSize;
        }
      }
      
      // update x values
      for (int n = 0; n < k; n++) {
        x[n] += stepSize * r[n];
        if (x[n] < EPSILON) {
          x[n] = 0.0;
        }
      }
      
      // TODO: do something in case of divergence
    }
    
    return x;
  }
  
}
