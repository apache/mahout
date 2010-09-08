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

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.random.Exponential;
import org.junit.Test;

import java.util.Random;

public final class AdaptiveLogisticRegressionTest extends MahoutTestCase {

  @Test
  public void testTrain() {

    Random gen = RandomUtils.getRandom();
    Exponential exp = new Exponential(0.5, gen);
    Vector beta = new DenseVector(200);
    for (Vector.Element element : beta) {
      int sign = 1;
      if (gen.nextDouble() < 0.5) {
        sign = -1;
      }
      element.set(sign * exp.nextDouble());
    }

    AdaptiveLogisticRegression.Wrapper cl = new AdaptiveLogisticRegression.Wrapper(2, 200, new L1());
    cl.update(new double[]{1.0e-5, 1});

    for (int i = 0; i < 10000; i++) {
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      cl.train(r);
      if (i % 1000 == 0) {
//        cl.close();
        System.out.printf("%10d %10.3f\n", i, cl.getLearner().auc());
      }
    }
    assertEquals(1, cl.getLearner().auc(), 0.1);

    AdaptiveLogisticRegression x = new AdaptiveLogisticRegression(2, 200, new L1());
    x.setInterval(1000);

    for (int i = 0; i < 20000; i++) {
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      x.train(r.getKey(), r.getActual(), r.getInstance());
      if (i % 1000 == 0) {
        if (x.getBest() != null) {
          System.out.printf("%10d %10.4f %10.8f %.3f\n",
                            i, x.auc(),
                            Math.log10(x.getBest().getMappedParams()[0]), x.getBest().getMappedParams()[1]);
        }
      }
    }
    assertEquals(1, x.auc(), 0.1);
  }

  private static AdaptiveLogisticRegression.TrainingExample getExample(int i, Random gen, Vector beta) {
    Vector data = new DenseVector(200);

    for (Vector.Element element : data) {
      element.set(gen.nextDouble() < 0.3 ? 1 : 0);
    }

    double p = 1 / (1 + Math.exp(1.5 - data.dot(beta)));
    int target = 0;
    if (gen.nextDouble() < p) {
      target = 1;
    }
    return new AdaptiveLogisticRegression.TrainingExample(i, target, data);
  }

  @Test
  public void copyLearnsAsExpected() {
    Random gen = RandomUtils.getRandom();
    Exponential exp = new Exponential(0.5, gen);
    Vector beta = new DenseVector(200);
    for (Vector.Element element : beta) {
        int sign = 1;
        if (gen.nextDouble() < 0.5) {
          sign = -1;
        }
      element.set(sign * exp.nextDouble());
    }

    // train one copy of a wrapped learner
    AdaptiveLogisticRegression.Wrapper w = new AdaptiveLogisticRegression.Wrapper(2, 200, new L1());
    for (int i = 0; i < 3000; i++) {
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      w.train(r);
      if (i % 1000 == 0) {
        System.out.printf("%10d %.3f\n", i, w.getLearner().auc());
      }
    }
    System.out.printf("%10d %.3f\n", 3000, w.getLearner().auc());
    double auc1 = w.getLearner().auc();

    // then switch to a copy of that learner ... progress should continue
    AdaptiveLogisticRegression.Wrapper w2 = w.copy();

    for (int i = 0; i < 5000; i++) {
      if (i % 1000 == 0) {
        if (i == 0) {
          assertEquals("Should have started with no data", 0.5, w2.getLearner().auc(), 0.0001);
        }
        if (i == 1000) {
          double auc2 = w2.getLearner().auc();
          assertTrue("Should have had head-start", Math.abs(auc2 - 0.5) > 0.1);
          assertTrue("AUC should improve quickly on copy", auc1 < auc2);
        }
        System.out.printf("%10d %.3f\n", i, w2.getLearner().auc());
      }
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      w2.train(r);
    }
    assertEquals("Original should not change after copy is updated", auc1, w.getLearner().auc(), 1.0e-5);

    // this improvement is really quite lenient
    assertTrue("AUC should improve significantly on copy", auc1 < w2.getLearner().auc() - 0.05);

    // make sure that the copy didn't lose anything
    assertEquals(auc1, w.getLearner().auc(), 0);
  }
}
