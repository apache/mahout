package org.apache.mahout.math.stats;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class OnlineAucTest {
  @Test
  public void binaryCase() {
    OnlineAuc a1 = new OnlineAuc();
    a1.setRandom(new Random(1));
    a1.setPolicy(OnlineAuc.ReplacementPolicy.FAIR);

    OnlineAuc a2 = new OnlineAuc();
    a2.setRandom(new Random(2));
    a2.setPolicy(OnlineAuc.ReplacementPolicy.FIFO);

    OnlineAuc a3 = new OnlineAuc();
    a3.setRandom(new Random(3));
    a3.setPolicy(OnlineAuc.ReplacementPolicy.RANDOM);

    Random gen = new Random(100);
    for (int i = 0; i < 10000; i++) {
      double x = gen.nextGaussian();

      a1.addSample(0, x);
      a2.addSample(0, x);
      a3.addSample(0, x);

      x = gen.nextGaussian() + 1;

      a1.addSample(1, x);
      a2.addSample(1, x);
      a3.addSample(1, x);
    }

    a1 = new OnlineAuc();
    a1.setPolicy(OnlineAuc.ReplacementPolicy.FAIR);

    a2 = new OnlineAuc();
    a2.setPolicy(OnlineAuc.ReplacementPolicy.FIFO);

    a3 = new OnlineAuc();
    a3.setPolicy(OnlineAuc.ReplacementPolicy.RANDOM);

    gen = new Random(1);
    for (int i = 0; i < 10000; i++) {
      double x = gen.nextGaussian();

      a1.addSample(1, x);
      a2.addSample(1, x);
      a3.addSample(1, x);

      x = gen.nextGaussian() + 1;

      a1.addSample(0, x);
      a2.addSample(0, x);
      a3.addSample(0, x);
    }

    // reference value computed using R: mean(rnorm(1000000) < rnorm(1000000,1))
    Assert.assertEquals(1 - 0.76, a1.auc(), 0.05);
    Assert.assertEquals(1 - 0.76, a2.auc(), 0.05);
    Assert.assertEquals(1 - 0.76, a3.auc(), 0.05);
  }
}
