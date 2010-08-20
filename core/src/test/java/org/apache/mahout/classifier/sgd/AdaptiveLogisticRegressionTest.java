package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.jet.random.Exponential;
import org.apache.mahout.math.jet.random.Normal;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;
import org.junit.Assert;
import org.junit.Test;

public class AdaptiveLogisticRegressionTest {
  @Test
  public void testTrain() {
    // we make up data for a simple model

    final MersenneTwister gen = new MersenneTwister(1);
    final Exponential exp = new Exponential(.5, gen);
    Vector data = new DenseVector(200);
    Vector beta = new DenseVector(200);
    for (Vector.Element element : beta) {
        int sign = 1;
        if (gen.nextDouble() < 0.5) {
          sign = -1;
        }
      element.set(sign * exp.nextDouble());
    }

    AdaptiveLogisticRegression.Wrapper cl = new AdaptiveLogisticRegression.Wrapper(2, 200, new L1());
    cl.update(new double[]{1e-5, 1});

    for (int i = 0; i < 10000; i++) {
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      cl.train(r);
      if (i % 1000 == 0) {
//        cl.close();
        System.out.printf("%10d %10.3f\n", i, cl.getLearner().auc());
      }
    }

    AdaptiveLogisticRegression x = new AdaptiveLogisticRegression(2, 200, new L1());
    x.setInterval(1000);

    final Normal norm = new Normal(0, 1, gen);
    for (int i = 0; i < 20000; i++) {
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      x.train(r.getKey(), r.getActual(), r.getInstance());
      if (i % 1000 == 0) {
        if (x.getBest() != null) {
          System.out.printf("%10d %10.4f %10.8f %.3f\n", i, x.auc(), Math.log10(x.getBest().getMappedParams()[0]), x.getBest().getMappedParams()[1]);
        }
      }
    }
  }

  private AdaptiveLogisticRegression.TrainingExample getExample(int i, MersenneTwister gen, Vector beta) {
    Vector data = new DenseVector(200);

    for (Vector.Element element : data) {
      element.set(gen.nextDouble() < 0.3 ? 1 : 0);
    }

    double p = 1 / (1 + Math.exp(1.5 - data.dot(beta)));
    int target = 0;
    if (gen.nextDouble() < p) {
      target = 1;
    }
    AdaptiveLogisticRegression.TrainingExample r = new AdaptiveLogisticRegression.TrainingExample(i, target, data);
    return r;
  }

  @Test
  public void copyLearnsAsExpected() {
    final MersenneTwister gen = new MersenneTwister(1);
    final Exponential exp = new Exponential(.5, gen);
    Vector data = new DenseVector(200);
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
    double auc2 = -1;

    for (int i = 0; i < 5000; i++) {
      if (i % 1000 == 0) {
        if (i == 0) {
          Assert.assertEquals("Should have started with no data", 0.5, w2.getLearner().auc(), 0.0001);
        }
        if (i == 1000) {
          auc2 = w2.getLearner().auc();
          Assert.assertTrue("Should have had head-start", Math.abs(auc2 - 0.5) > 0.1);
        }
        System.out.printf("%10d %.3f\n", i, w2.getLearner().auc());
      }
      AdaptiveLogisticRegression.TrainingExample r = getExample(i, gen, beta);
      w2.train(r);
    }
    Assert.assertTrue("AUC should improve on copy", auc1 < w2.getLearner().auc() - 0.1);
    Assert.assertTrue("AUC should improve on copy", auc1 < auc2);

    // make sure that the copy didn't lose anything
    Assert.assertEquals(auc1, w.getLearner().auc(), 0);
  }
}
