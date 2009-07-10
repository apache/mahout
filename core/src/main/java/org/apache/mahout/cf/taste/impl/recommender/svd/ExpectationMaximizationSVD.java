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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import org.apache.mahout.cf.taste.impl.common.RandomUtils;

import java.util.Random;

/** Calculates the SVD using an Expectation Maximization algorithm. */
public final class ExpectationMaximizationSVD {

  private static final Random random = RandomUtils.getRandom();

  private static final double LEARNING_RATE = 0.005;
  /** Parameter used to prevent overfitting. 0.02 is a good value. */
  private static final double K = 0.02;
  /** Random noise applied to starting values. */
  private static final double r = 0.005;

  private final int m;
  private final int n;
  private final int k;

  /** User singular vector. */
  private final double[][] leftVector;

  /** Item singular vector. */
  private final double[][] rightVector;

  /**
   * @param m            number of columns
   * @param n            number of rows
   * @param k            number of features
   * @param defaultValue default starting values for the SVD vectors
   */
  public ExpectationMaximizationSVD(int m, int n, int k, double defaultValue) {
    this(m, n, k, defaultValue, r);
  }

  public ExpectationMaximizationSVD(int m, int n, int k, double defaultValue, double noise) {
    this.m = m;
    this.n = n;
    this.k = k;

    leftVector = new double[m][k];
    rightVector = new double[n][k];

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < m; j++) {
        leftVector[j][i] = defaultValue + (random.nextDouble() - 0.5) * noise;
      }
      for (int j = 0; j < n; j++) {
        rightVector[j][i] = defaultValue + (random.nextDouble() - 0.5) * noise;
      }
    }
  }

  public double getDotProduct(int i, int j) {
    double result = 1.0;
    for (int k = 0; k < this.k; k++) {
      result += leftVector[i][k] * rightVector[j][k];
    }
    return result;
  }

  public void train(int i, int j, int k, double value) {
    double err = value - getDotProduct(i, j);
    double[] leftVectorI = leftVector[i];
    double[] rightVectorJ = rightVector[j];
    leftVectorI[k] += LEARNING_RATE * (err * rightVectorJ[k] - K * leftVectorI[k]);
    rightVectorJ[k] += LEARNING_RATE * (err * leftVectorI[k] - K * rightVectorJ[k]);
  }

  int getM() {
    return m;
  }

  int getN() {
    return n;
  }

  int getK() {
    return k;
  }

}
