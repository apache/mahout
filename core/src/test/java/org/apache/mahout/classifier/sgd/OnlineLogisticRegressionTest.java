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

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.CharStreams;
import com.google.common.io.Resources;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class OnlineLogisticRegressionTest extends MahoutTestCase {

  private Matrix input;

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


    train(input, target, lr);

    System.out.printf("%.2f %.5f\n", lr.auc(), lr.logLikelihood());
    test(input, target, lr);

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

    train(input, target, lr);
    test(input, target, lr);
  }

  private Vector readStandardData() throws IOException {
    // 60 test samples.  First column is constant.  Second and third are normally distributed from
    // either N([2,2], 1) (rows 0...29) or N([-2,-2], 1) (rows 30...59).  The first 30 rows have a
    // target variable of 0, the last 30 a target of 1.  The remaining columns are are random noise.
    input = readCsv("sgd.csv");

    // regenerate the target variable
    Vector target = new DenseVector(60);
    target.assign(0);
    target.viewPart(30, 30).assign(1);
    return target;
  }

  private static void train(Matrix input, Vector target, OnlineLearner lr) {
    RandomUtils.useTestSeed();
    Random gen = RandomUtils.getRandom();

    // train on samples in random order (but only one pass)
    for (int row : permute(gen, 60)) {
      lr.train((int) target.get(row), input.getRow(row));
    }
    lr.close();
  }

  private static void test(Matrix input, Vector target, AbstractVectorClassifier lr) {
    // now test the accuracy
    Matrix tmp = lr.classify(input);
    // mean(abs(tmp - target))
    double meanAbsoluteError = tmp.getColumn(0).minus(target).aggregate(Functions.PLUS, Functions.ABS) / 60;

    // max(abs(tmp - target)
    double maxAbsoluteError = tmp.getColumn(0).minus(target).aggregate(Functions.MAX, Functions.ABS);

    System.out.printf("mAE = %.4f, maxAE = %.4f\n", meanAbsoluteError, maxAbsoluteError);
    assertEquals(0, meanAbsoluteError , 0.05);
    assertEquals(0, maxAbsoluteError, 0.3);

    // convenience methods should give the same results
    Vector v = lr.classifyScalar(input);
    assertEquals(0, v.minus(tmp.getColumn(0)).norm(1), 1.0e-5);
    v = lr.classifyFull(input).getColumn(1);
    assertEquals(0, v.minus(tmp.getColumn(0)).norm(1), 1.0e-4);
  }

  /**
   * Permute the integers from 0 ... max-1
   *
   * @param gen The random number generator to use.
   * @param max The number of integers to permute
   * @return An array of jumbled integer values
   */
  private static int[] permute(Random gen, int max) {
    int[] permutation = new int[max];
    permutation[0] = 0;
    for (int i = 1; i < max; i++) {
      int n = gen.nextInt(i + 1);
      if (n == i) {
        permutation[i] = i;
      } else {
        permutation[i] = permutation[n];
        permutation[n] = i;
      }
    }
    return permutation;
  }


  /**
   * Reads a file containing CSV data.  This isn't implemented quite the way you might like for a
   * real program, but does the job for reading test data.  Most notably, it will only read numbers,
   * not quoted strings.
   *
   * @param resourceName Where to get the data.
   * @return A matrix of the results.
   * @throws IOException If there is an error reading the data
   */
  private static Matrix readCsv(String resourceName) throws IOException {
    Splitter onCommas = Splitter.on(",").trimResults(CharMatcher.anyOf(" \""));

    Readable isr = new InputStreamReader(Resources.getResource(resourceName).openStream(), Charsets.UTF_8);
    List<String> data = CharStreams.readLines(isr);
    String first = data.get(0);
    data = data.subList(1, data.size());

    List<String> values = Lists.newArrayList(onCommas.split(first));
    Matrix r = new DenseMatrix(data.size(), values.size());

    int column = 0;
    Map<String, Integer> labels = Maps.newHashMap();
    for (String value : values) {
      labels.put(value, column);
      column++;
    }
    r.setColumnLabelBindings(labels);

    int row = 0;
    for (String line : data) {
      column = 0;
      values = Lists.newArrayList(onCommas.split(line));
      for (String value : values) {
        r.set(row, column, Double.parseDouble(value));
        column++;
      }
      row++;
    }

    return r;
  }
}
