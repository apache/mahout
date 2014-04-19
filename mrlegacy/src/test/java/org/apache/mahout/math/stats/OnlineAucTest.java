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

package org.apache.mahout.math.stats;

import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Random;

public final class OnlineAucTest extends MahoutTestCase {

  @Test
  public void testBinaryCase() {
    Random gen = RandomUtils.getRandom();

    OnlineSummarizer[] stats = new OnlineSummarizer[4];
    for (int i = 0; i < 4; i++) {
      stats[i] = new OnlineSummarizer();
    }

    for (int i = 0; i < 100; i++) {
      OnlineAuc a1 = new GlobalOnlineAuc();
      a1.setPolicy(GlobalOnlineAuc.ReplacementPolicy.FAIR);

      OnlineAuc a2 = new GlobalOnlineAuc();
      a2.setPolicy(GlobalOnlineAuc.ReplacementPolicy.FIFO);

      OnlineAuc a3 = new GlobalOnlineAuc();
      a3.setPolicy(GlobalOnlineAuc.ReplacementPolicy.RANDOM);

      Auc a4 = new Auc();

      for (int j = 0; j < 10000; j++) {
        double x = gen.nextGaussian();

        a1.addSample(0, x);
        a2.addSample(0, x);
        a3.addSample(0, x);
        a4.add(0, x);

        x = gen.nextGaussian() + 1;

        a1.addSample(1, x);
        a2.addSample(1, x);
        a3.addSample(1, x);
        a4.add(1, x);
      }

      stats[0].add(a1.auc());
      stats[1].add(a2.auc());
      stats[2].add(a3.auc());
      stats[3].add(a4.auc());
    }
    
    int i = 0;
    for (GlobalOnlineAuc.ReplacementPolicy policy : new GlobalOnlineAuc.ReplacementPolicy[] {
                                                      GlobalOnlineAuc.ReplacementPolicy.FAIR,
                                                      GlobalOnlineAuc.ReplacementPolicy.FIFO,
                                                      GlobalOnlineAuc.ReplacementPolicy.RANDOM,
                                                      null}) {
      OnlineSummarizer summary = stats[i++];
      System.out.printf("%s,%.4f (min = %.4f, 25%%-ile=%.4f, 75%%-ile=%.4f, max=%.4f)\n", policy, summary.getMean(),
        summary.getQuartile(0), summary.getQuartile(1), summary.getQuartile(2), summary.getQuartile(3));

    }

    // FAIR policy isn't so accurate
    assertEquals(0.7603, stats[0].getMean(), 0.03);
    assertEquals(0.7603, stats[0].getQuartile(1), 0.03);
    assertEquals(0.7603, stats[0].getQuartile(3), 0.03);

    // FIFO policy seems best
    assertEquals(0.7603, stats[1].getMean(), 0.001);
    assertEquals(0.7603, stats[1].getQuartile(1), 0.006);
    assertEquals(0.7603, stats[1].getQuartile(3), 0.006);

    // RANDOM policy is nearly the same as FIFO
    assertEquals(0.7603, stats[2].getMean(), 0.001);
    assertEquals(0.7603, stats[2].getQuartile(1), 0.006);
    assertEquals(0.7603, stats[2].getQuartile(1), 0.006);
  }

  @Test(expected=UnsupportedOperationException.class)
  public void mustNotOmitGroup() {
    OnlineAuc x = new GroupedOnlineAuc();
    x.addSample(0, 3.14);
  }

  @Test
  public void groupedAuc() {
    Random gen = RandomUtils.getRandom();
    OnlineAuc x = new GroupedOnlineAuc();
    OnlineAuc y = new GlobalOnlineAuc();

    for (int i = 0; i < 10000; i++) {
      x.addSample(0, "a", gen.nextGaussian());
      x.addSample(1, "a", gen.nextGaussian() + 1);
      x.addSample(0, "b", gen.nextGaussian() + 10);
      x.addSample(1, "b", gen.nextGaussian() + 11);

      y.addSample(0, "a", gen.nextGaussian());
      y.addSample(1, "a", gen.nextGaussian() + 1);
      y.addSample(0, "b", gen.nextGaussian() + 10);
      y.addSample(1, "b", gen.nextGaussian() + 11);
    }

    assertEquals(0.7603, x.auc(), 0.01);
    assertEquals((0.7603 + 0.5) / 2, y.auc(), 0.02);
  }
}
