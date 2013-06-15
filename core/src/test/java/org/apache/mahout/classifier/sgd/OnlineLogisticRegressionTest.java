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

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.Closer;
import com.google.common.io.Resources;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public final class OnlineLogisticRegressionTest extends OnlineBaseTest {

  private static final Logger logger = LoggerFactory.getLogger(OnlineLogisticRegressionTest.class);

  /**
   * The CrossFoldLearner is probably the best learner to use for new applications.
   *
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
  public void iris() throws IOException {
    // this test trains a 3-way classifier on the famous Iris dataset.
    // a similar exercise can be accomplished in R using this code:
    //    library(nnet)
    //    correct = rep(0,100)
    //    for (j in 1:100) {
    //      i = order(runif(150))
    //      train = iris[i[1:100],]
    //      test = iris[i[101:150],]
    //      m = multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, train)
    //      correct[j] = mean(predict(m, newdata=test) == test$Species)
    //    }
    //    hist(correct)
    //
    // Note that depending on the training/test split, performance can be better or worse.
    // There is about a 5% chance of getting accuracy < 90% and about 20% chance of getting accuracy
    // of 100%
    //
    // This test uses a deterministic split that is neither outstandingly good nor bad


    RandomUtils.useTestSeed();
    Splitter onComma = Splitter.on(",");

    // read the data
    List<String> raw = Resources.readLines(Resources.getResource("iris.csv"), Charsets.UTF_8);

    // holds features
    List<Vector> data = Lists.newArrayList();

    // holds target variable
    List<Integer> target = Lists.newArrayList();

    // for decoding target values
    Dictionary dict = new Dictionary();

    // for permuting data later
    List<Integer> order = Lists.newArrayList();

    for (String line : raw.subList(1, raw.size())) {
      // order gets a list of indexes
      order.add(order.size());

      // parse the predictor variables
      Vector v = new DenseVector(5);
      v.set(0, 1);
      int i = 1;
      Iterable<String> values = onComma.split(line);
      for (String value : Iterables.limit(values, 4)) {
        v.set(i++, Double.parseDouble(value));
      }
      data.add(v);

      // and the target
      target.add(dict.intern(Iterables.get(values, 4)));
    }

    // randomize the order ... original data has each species all together
    // note that this randomization is deterministic
    Random random = RandomUtils.getRandom();
    Collections.shuffle(order, random);

    // select training and test data
    List<Integer> train = order.subList(0, 100);
    List<Integer> test = order.subList(100, 150);
    logger.warn("Training set = {}", train);
    logger.warn("Test set = {}", test);

    // now train many times and collect information on accuracy each time
    int[] correct = new int[test.size() + 1];
    for (int run = 0; run < 200; run++) {
      OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 5, new L2(1));
      // 30 training passes should converge to > 95% accuracy nearly always but never to 100%
      for (int pass = 0; pass < 30; pass++) {
        Collections.shuffle(train, random);
        for (int k : train) {
          lr.train(target.get(k), data.get(k));
        }
      }

      // check the accuracy on held out data
      int x = 0;
      int[] count = new int[3];
      for (Integer k : test) {
        int r = lr.classifyFull(data.get(k)).maxValueIndex();
        count[r]++;
        x += r == target.get(k) ? 1 : 0;
      }
      correct[x]++;
    }

    // verify we never saw worse than 95% correct,
    for (int i = 0; i < Math.floor(0.95 * test.size()); i++) {
      assertEquals(String.format("%d trials had unacceptable accuracy of only %.0f%%: ", correct[i], 100.0 * i / test.size()), 0, correct[i]);
    }
    // nor perfect
    assertEquals(String.format("%d trials had unrealistic accuracy of 100%%", correct[test.size() - 1]), 0, correct[test.size()]);
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

  /**
   * Test for Serialization/DeSerialization
   *
   */
  @Test
  public void testSerializationAndDeSerialization() throws Exception {
    OnlineLogisticRegression lr = new OnlineLogisticRegression(2, 8, new L1())
      .lambda(1 * 1.0e-3)
      .stepOffset(11)
      .alpha(0.01)
      .learningRate(50)
      .decayExponent(-0.02);

    lr.close();

    byte[] output;
    Closer closer = Closer.create();

    try {
      ByteArrayOutputStream byteArrayOutputStream = closer.register(new ByteArrayOutputStream());
      DataOutputStream dataOutputStream = closer.register(new DataOutputStream(byteArrayOutputStream));
      PolymorphicWritable.write(dataOutputStream, lr);
      output = byteArrayOutputStream.toByteArray();
    } finally {
      closer.close();
    }

    OnlineLogisticRegression read;

    try {
      ByteArrayInputStream byteArrayInputStream = closer.register(new ByteArrayInputStream(output));
      DataInputStream dataInputStream = closer.register(new DataInputStream(byteArrayInputStream));
      read = closer.register(PolymorphicWritable.read(dataInputStream, OnlineLogisticRegression.class));
    } finally {
      closer.close();
    }

    //lambda
    Assert.assertEquals((1.0e-3), read.getLambda(), 1.0e-7);

    // Reflection to get private variables
    //stepOffset
    Field stepOffset = lr.getClass().getDeclaredField("stepOffset");
    stepOffset.setAccessible(true);
    int stepOffsetVal = (Integer) stepOffset.get(lr);
    Assert.assertEquals(11, stepOffsetVal);

    //decayFactor (alpha)
    Field decayFactor = lr.getClass().getDeclaredField("decayFactor");
    decayFactor.setAccessible(true);
    double decayFactorVal = (Double) decayFactor.get(lr);
    Assert.assertEquals(0.01, decayFactorVal, 1.0e-7);

    //learning rate (mu0)
    Field mu0 = lr.getClass().getDeclaredField("mu0");
    mu0.setAccessible(true);
    double mu0Val = (Double) mu0.get(lr);
    Assert.assertEquals(50, mu0Val, 1.0e-7);

    //forgettingExponent (decayExponent)
    Field forgettingExponent = lr.getClass().getDeclaredField("forgettingExponent");
    forgettingExponent.setAccessible(true);
    double forgettingExponentVal = (Double) forgettingExponent.get(lr);
    Assert.assertEquals(-0.02, forgettingExponentVal, 1.0e-7);
  }
}