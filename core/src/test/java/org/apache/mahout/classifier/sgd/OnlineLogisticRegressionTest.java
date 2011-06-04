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

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

public final class OnlineLogisticRegressionTest extends OnlineBaseTest {

  /**
   * The CrossFoldLearner is probably the best learner to use for new applications.
    * @throws IOException If test resources aren't readable.
   */
  @Test
  public void crossValidation() throws IOException {
    Vector target = readStandardData();

    CrossFoldLearner lr = new CrossFoldLearner(5, 2, 8, new L1())
            .lambda(1 * 1.0e-3)
            .learningRate(50);


    train(getInput(), target, lr);

    System.out.printf("%.2f %.5f\n", lr.auc(), lr.logLikelihood());
    test(getInput(), target, lr, 0.05, 0.3);

  }

  @Test
  public void crossValidatedAuc() throws IOException {
    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();

    Matrix data = readCsv("cancer.csv");
    CrossFoldLearner lr = new CrossFoldLearner(5, 2, 10, new L1())
            .stepOffset(10)
            .decayExponent(0.7)
            .lambda(1 * 1.0e-3)
            .learningRate(5);
    int k = 0;
    int[] ordering = permute(gen, data.numRows());
    for (int epoch = 0; epoch < 100; epoch++) {
      for (int row : ordering) {
        lr.train(row, (int) data.get(row, 9), data.viewRow(row));
        System.out.printf("%d,%d,%.3f\n", epoch, k++, lr.auc());
      }
      assertEquals(1, lr.auc(), 0.2);
    }
    assertEquals(1, lr.auc(), 0.1);
  }

  /**
   * Verifies that a classifier with known coefficients does the right thing.
   */
  @Test
  public void testClassify() {
    OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 2, new L2(1));
    // set up some internal coefficients as if we had learned them
    lr.setBeta(0, 0, -1);
    lr.setBeta(1, 0, -2);

    // zero vector gives no information.  All classes are equal.
    Vector v = lr.classify(new DenseVector(new double[]{0, 0}));
    assertEquals(1 / 3.0, v.get(0), 1.0e-8);
    assertEquals(1 / 3.0, v.get(1), 1.0e-8);

    v = lr.classifyFull(new DenseVector(new double[]{0, 0}));
    assertEquals(1.0, v.zSum(), 1.0e-8);
    assertEquals(1 / 3.0, v.get(0), 1.0e-8);
    assertEquals(1 / 3.0, v.get(1), 1.0e-8);
    assertEquals(1 / 3.0, v.get(2), 1.0e-8);

    // weights for second vector component are still zero so all classifications are equally likely
    v = lr.classify(new DenseVector(new double[]{0, 1}));
    assertEquals(1 / 3.0, v.get(0), 1.0e-3);
    assertEquals(1 / 3.0, v.get(1), 1.0e-3);

    v = lr.classifyFull(new DenseVector(new double[]{0, 1}));
    assertEquals(1.0, v.zSum(), 1.0e-8);
    assertEquals(1 / 3.0, v.get(0), 1.0e-3);
    assertEquals(1 / 3.0, v.get(1), 1.0e-3);
    assertEquals(1 / 3.0, v.get(2), 1.0e-3);

    // but the weights on the first component are non-zero
    v = lr.classify(new DenseVector(new double[]{1, 0}));
    assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1.0e-8);
    assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1.0e-8);

    v = lr.classifyFull(new DenseVector(new double[]{1, 0}));
    assertEquals(1.0, v.zSum(), 1.0e-8);
    assertEquals(1 / (1 + Math.exp(-1) + Math.exp(-2)), v.get(0), 1.0e-8);
    assertEquals(Math.exp(-1) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(1), 1.0e-8);
    assertEquals(Math.exp(-2) / (1 + Math.exp(-1) + Math.exp(-2)), v.get(2), 1.0e-8);

    lr.setBeta(0, 1, 1);

    v = lr.classifyFull(new DenseVector(new double[]{1, 1}));
    assertEquals(1.0, v.zSum(), 1.0e-8);
    assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(-2)), v.get(1), 1.0e-3);
    assertEquals(Math.exp(-2) / (1 + Math.exp(0) + Math.exp(-2)), v.get(2), 1.0e-3);
    assertEquals(1 / (1 + Math.exp(0) + Math.exp(-2)), v.get(0), 1.0e-3);

    lr.setBeta(1, 1, 3);

    v = lr.classifyFull(new DenseVector(new double[]{1, 1}));
    assertEquals(1.0, v.zSum(), 1.0e-8);
    assertEquals(Math.exp(0) / (1 + Math.exp(0) + Math.exp(1)), v.get(1), 1.0e-8);
    assertEquals(Math.exp(1) / (1 + Math.exp(0) + Math.exp(1)), v.get(2), 1.0e-8);
    assertEquals(1 / (1 + Math.exp(0) + Math.exp(1)), v.get(0), 1.0e-8);
  }

  @Test
  public void testTrain() throws Exception {
    Vector target = readStandardData();


    // lambda here needs to be relatively small to avoid swamping the actual signal, but can be
    // larger than usual because the data are dense.  The learning rate doesn't matter too much
    // for this example, but should generally be < 1
    // --passes 1 --rate 50 --lambda 0.001 --input sgd-y.csv --features 21 --output model --noBias
    //   --target y --categories 2 --predictors  V2 V3 V4 V5 V6 V7 --types n
    OnlineLogisticRegression lr = new OnlineLogisticRegression(2, 8, new L1())
            .lambda(1 * 1.0e-3)
            .learningRate(50);

    train(getInput(), target, lr);
    test(getInput(), target, lr, 0.05, 0.3);
  }

}