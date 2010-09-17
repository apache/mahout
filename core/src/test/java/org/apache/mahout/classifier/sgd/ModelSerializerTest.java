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

import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.ep.Mapping;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.math.stats.OnlineAuc;
import org.junit.Test;

import java.io.StringReader;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public final class ModelSerializerTest extends MahoutTestCase {

  @Test
  public void testSoftLimitDeserialization() {
    Mapping m = ModelSerializer.gson().fromJson(new StringReader("{\"min\":-18.420680743952367,\"max\":-2.3025850929940455,\"scale\":1.0}"), Mapping.SoftLimit.class);
    assertTrue(m instanceof Mapping.SoftLimit);
    assertEquals((-18.420680743952367 + -2.3025850929940455) / 2, m.apply(0), 1.0e-6);

    String data = "{\"class\":\"org.apache.mahout.ep.Mapping$SoftLimit\",\"value\":{\"min\":-18.420680743952367,\"max\":-2.3025850929940455,\"scale\":1.0}}";
    m = ModelSerializer.gson().fromJson(new StringReader(data), Mapping.class);
    assertTrue(m instanceof Mapping.SoftLimit);
    assertEquals((-18.420680743952367 + -2.3025850929940455) / 2, m.apply(0), 1.0e-6);
  }

  @Test
  public void testMappingDeserialization() {
    String data = "{\"class\":\"org.apache.mahout.ep.Mapping$LogLimit\",\"value\":{\"wrapped\":{\"class\":\"org.apache.mahout.ep.Mapping$SoftLimit\",\"value\":{\"min\":-18.420680743952367,\"max\":-2.3025850929940455,\"scale\":1.0}}}}";
    Mapping m = ModelSerializer.gson().fromJson(new StringReader(data), Mapping.class);
    assertTrue(m instanceof Mapping.LogLimit);
    assertEquals(Math.sqrt(Math.exp(-18.420680743952367) * Math.exp(-2.3025850929940455)), m.apply(0), 1.0e-6);
  }

  @Test
  public void onlineAucRoundtrip() {
    RandomUtils.useTestSeed();
    OnlineAuc auc1 = new OnlineAuc();
    Random gen = new Random(2);
    for (int i = 0; i < 10000; i++) {
      auc1.addSample(0, gen.nextGaussian());
      auc1.addSample(1, gen.nextGaussian() + 1);
    }
    assertEquals(0.76, auc1.auc(), 0.04);

    Gson gson = ModelSerializer.gson();
    String s = gson.toJson(auc1);

    OnlineAuc auc2 = gson.fromJson(s, OnlineAuc.class);

    assertEquals(auc1.auc(), auc2.auc(), 0);

    for (int i = 0; i < 1000; i++) {
      auc1.addSample(0, gen.nextGaussian());
      auc1.addSample(1, gen.nextGaussian() + 1);

      auc2.addSample(0, gen.nextGaussian());
      auc2.addSample(1, gen.nextGaussian() + 1);
    }

    assertEquals(auc1.auc(), auc2.auc(), 0.01);
  }

  @Test
  public void onlineLogisticRegressionRoundTrip() {
    OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 5, new L1());
    train(olr, 100);
    Gson gson = ModelSerializer.gson();
    String s = gson.toJson(olr);
    OnlineLogisticRegression olr2 = gson.fromJson(new StringReader(s), OnlineLogisticRegression.class);
    assertEquals(0, olr.getBeta().minus(olr2.getBeta()).aggregate(Functions.MAX, Functions.IDENTITY), 1.0e-6);

    train(olr, 100);
    train(olr2, 100);

    assertEquals(0, olr.getBeta().minus(olr2.getBeta()).aggregate(Functions.MAX, Functions.IDENTITY), 1.0e-6);
  }

  @Test
  public void crossFoldLearnerRoundTrip() {
    CrossFoldLearner learner = new CrossFoldLearner(5, 2, 5, new L1());
    train(learner, 100);
    Gson gson = ModelSerializer.gson();
    String s = gson.toJson(learner);
    CrossFoldLearner olr2 = gson.fromJson(new StringReader(s), CrossFoldLearner.class);
    double auc1 = learner.auc();
    assertTrue(auc1 > 0.85);
    assertEquals(auc1, olr2.auc(), 1.0e-6);

    train(learner, 100);
    train(olr2, 100);

    assertEquals(learner.auc(), olr2.auc(), 0.02);
    double auc2 = learner.auc();
    assertTrue(auc2 > auc1);
  }

  @Test
  public void adaptiveLogisticRegressionRoundTrip() {
    AdaptiveLogisticRegression learner = new AdaptiveLogisticRegression(2, 5, new L1());
    learner.setInterval(200);
    train(learner, 400);
    Gson gson = ModelSerializer.gson();
    String s = gson.toJson(learner);
    AdaptiveLogisticRegression olr2 = gson.fromJson(new StringReader(s), AdaptiveLogisticRegression.class);
    double auc1 = learner.auc();
    assertTrue(auc1 > 0.85);
    assertEquals(auc1, olr2.auc(), 1.0e-6);

    train(learner, 1000);
    train(olr2, 1000);

    assertEquals(learner.auc(), olr2.auc(), 0.005);
    double auc2 = learner.auc();
    assertTrue(String.format("%.3f > %.3f", auc2, auc1), auc2 > auc1);
  }

  @Test
  public void trainingExampleList() {
    Random gen = new Random(1);
    List<AdaptiveLogisticRegression.TrainingExample> x1 = Lists.newArrayList();
    for (int i = 0; i < 10; i++) {
      AdaptiveLogisticRegression.TrainingExample t =
          new AdaptiveLogisticRegression.TrainingExample(i, i % 3, randomVector(gen, 5));
      x1.add(t);
    }

    Gson gson = ModelSerializer.gson();
    Type listType = new TypeToken<List<AdaptiveLogisticRegression.TrainingExample>>() {
    }.getType();
    String s = gson.toJson(x1, listType);

    List<AdaptiveLogisticRegression.TrainingExample> x2 = gson.fromJson(new StringReader(s), listType);

    assertEquals(x1.size(), x2.size());
    Iterator<AdaptiveLogisticRegression.TrainingExample> it = x2.iterator();
    for (AdaptiveLogisticRegression.TrainingExample example : x1) {
      AdaptiveLogisticRegression.TrainingExample example2 = it.next();
      assertEquals(example.getKey(), example2.getKey());
      assertEquals(0, example.getInstance().minus(example2.getInstance()).maxValue(), 1.0e-6);
      assertEquals(example.getActual(), example2.getActual());
    }
  }

  private static void train(OnlineLearner olr, int n) {
    Vector beta = new DenseVector(new double[]{1, -1, 0, 0.5, -0.5});
    Random gen = new Random(1);
    for (int i = 0; i < n; i++) {
      Vector x = randomVector(gen, 5);

      int target = gen.nextDouble() < beta.dot(x) ? 1 : 0;
      olr.train(target, x);
    }
  }

  private static Vector randomVector(final Random gen, int n) {
    Vector x = new DenseVector(n);
    x.assign(new UnaryFunction() {
      @Override
      public double apply(double v) {
        return gen.nextGaussian();
      }
    });
    return x;
  }
}
