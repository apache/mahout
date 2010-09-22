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

import static org.apache.mahout.math.stats.OnlineAuc.ReplacementPolicy.*;

public final class OnlineAucTest extends MahoutTestCase {

  @Test
  public void testBinaryCase() {
    Random gen = RandomUtils.getRandom();

    OnlineSummarizer[] stats = new OnlineSummarizer[4];
    for (int i = 0; i < 4; i++) {
      stats[i] = new OnlineSummarizer();
    }

    for (int i = 0; i < 500; i++) {
      OnlineAuc a1 = new OnlineAuc();
      a1.setPolicy(FAIR);

      OnlineAuc a2 = new OnlineAuc();
      a2.setPolicy(FIFO);

      OnlineAuc a3 = new OnlineAuc();
      a3.setPolicy(RANDOM);

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
    for (OnlineAuc.ReplacementPolicy policy : new OnlineAuc.ReplacementPolicy[]{FAIR, FIFO, RANDOM, null}) {
      OnlineSummarizer summary = stats[i++];
      System.out.printf("%s,%.4f (min = %.4f, 25%%-ile=%.4f, 75%%-ile=%.4f, max=%.4f)\n", policy, summary.getMean(),
        summary.getQuartile(0), summary.getQuartile(1), summary.getQuartile(2), summary.getQuartile(3));

    }

    // FAIR policy isn't so accurate
    assertEquals(0.7603, stats[0].getMean(), 0.03);
    assertEquals(0.7603, stats[0].getQuartile(1), 0.025);
    assertEquals(0.7603, stats[0].getQuartile(3), 0.025);

    // FIFO policy seems best
    assertEquals(0.7603, stats[1].getMean(), 0.001);
    assertEquals(0.7603, stats[1].getQuartile(1), 0.006);
    assertEquals(0.7603, stats[1].getQuartile(3), 0.006);

    // RANDOM policy is nearly the same as FIFO
    assertEquals(0.7603, stats[2].getMean(), 0.001);
    assertEquals(0.7603, stats[2].getQuartile(1), 0.006);
    assertEquals(0.7603, stats[2].getQuartile(1), 0.006);
  }
}
