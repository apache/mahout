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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakLingering;
import com.google.common.io.Closeables;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.GlobalOnlineAuc;
import org.apache.mahout.math.stats.OnlineAuc;
import org.junit.Test;

public final class ModelSerializerTest extends MahoutTestCase {

  private static <T extends Writable> T roundTrip(T m, Class<T> clazz) throws IOException {
    ByteArrayOutputStream buf = new ByteArrayOutputStream(1000);
    DataOutputStream dos = new DataOutputStream(buf);
    try {
      PolymorphicWritable.write(dos, m);
    } finally {
      Closeables.close(dos, false);
    }
    return PolymorphicWritable.read(new DataInputStream(new ByteArrayInputStream(buf.toByteArray())), clazz);
  }

  @Test
  public void onlineAucRoundtrip() throws IOException {
    RandomUtils.useTestSeed();
    OnlineAuc auc1 = new GlobalOnlineAuc();
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < 10000; i++) {
      auc1.addSample(0, gen.nextGaussian());
      auc1.addSample(1, gen.nextGaussian() + 1);
    }
    assertEquals(0.76, auc1.auc(), 0.01);

    OnlineAuc auc3 = roundTrip(auc1, OnlineAuc.class);

    assertEquals(auc1.auc(), auc3.auc(), 0);

    for (int i = 0; i < 1000; i++) {
      auc1.addSample(0, gen.nextGaussian());
      auc1.addSample(1, gen.nextGaussian() + 1);

      auc3.addSample(0, gen.nextGaussian());
      auc3.addSample(1, gen.nextGaussian() + 1);
    }

    assertEquals(auc1.auc(), auc3.auc(), 0.01);
  }

  @Test
  public void onlineLogisticRegressionRoundTrip() throws IOException {
    OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 5, new L1());
    train(olr, 100);
    OnlineLogisticRegression olr3 = roundTrip(olr, OnlineLogisticRegression.class);
    assertEquals(0, olr.getBeta().minus(olr3.getBeta()).aggregate(Functions.MAX, Functions.IDENTITY), 1.0e-6);

    train(olr, 100);
    train(olr3, 100);

    assertEquals(0, olr.getBeta().minus(olr3.getBeta()).aggregate(Functions.MAX, Functions.IDENTITY), 1.0e-6);
    olr.close();
    olr3.close();
  }

  @Test
  public void crossFoldLearnerRoundTrip() throws IOException {
    CrossFoldLearner learner = new CrossFoldLearner(5, 2, 5, new L1());
    train(learner, 100);
    CrossFoldLearner olr3 = roundTrip(learner, CrossFoldLearner.class);
    double auc1 = learner.auc();
    assertTrue(auc1 > 0.85);
    assertEquals(auc1, learner.auc(), 1.0e-6);
    assertEquals(auc1, olr3.auc(), 1.0e-6);

    train(learner, 100);
    train(learner, 100);
    train(olr3, 100);

    assertEquals(learner.auc(), learner.auc(), 0.02);
    assertEquals(learner.auc(), olr3.auc(), 0.02);
    double auc2 = learner.auc();
    assertTrue(auc2 > auc1);
    learner.close();
    olr3.close();
  }

  @ThreadLeakLingering(linger = 1000)
  @Test
  public void adaptiveLogisticRegressionRoundTrip() throws IOException {
    AdaptiveLogisticRegression learner = new AdaptiveLogisticRegression(2, 5, new L1());
    learner.setInterval(200);
    train(learner, 400);
    AdaptiveLogisticRegression olr3 = roundTrip(learner, AdaptiveLogisticRegression.class);
    double auc1 = learner.auc();
    assertTrue(auc1 > 0.85);
    assertEquals(auc1, learner.auc(), 1.0e-6);
    assertEquals(auc1, olr3.auc(), 1.0e-6);

    train(learner, 1000);
    train(learner, 1000);
    train(olr3, 1000);

    assertEquals(learner.auc(), learner.auc(), 0.005);
    assertEquals(learner.auc(), olr3.auc(), 0.005);
    double auc2 = learner.auc();
    assertTrue(String.format("%.3f > %.3f", auc2, auc1), auc2 > auc1);
    learner.close();
    olr3.close();
  }

  private static void train(OnlineLearner olr, int n) {
    Vector beta = new DenseVector(new double[]{1, -1, 0, 0.5, -0.5});
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < n; i++) {
      Vector x = randomVector(gen, 5);

      int target = gen.nextDouble() < beta.dot(x) ? 1 : 0;
      olr.train(target, x);
    }
  }

  private static Vector randomVector(final Random gen, int n) {
    Vector x = new DenseVector(n);
    x.assign(new DoubleFunction() {
      @Override
      public double apply(double v) {
        return gen.nextGaussian();
      }
    });
    return x;
  }
}
