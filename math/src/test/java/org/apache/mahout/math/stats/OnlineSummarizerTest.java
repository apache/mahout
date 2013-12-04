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

package org.apache.mahout.math.stats;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.jet.random.AbstractContinousDistribution;
import org.apache.mahout.math.jet.random.Gamma;
import org.junit.Test;

import java.util.Random;

public final class OnlineSummarizerTest extends MahoutTestCase {

  @Test
  public void testCount() {
    OnlineSummarizer x = new OnlineSummarizer();
    assertEquals(0, x.getCount());
    x.add(1);
    assertEquals(1, x.getCount());

    for (int i = 2; i < 110; i++) {
      x.add(i);
      assertEquals(i, x.getCount());
    }
  }

  @Test
  public void testStats() {
    // the reference limits here were derived using a numerical simulation where I took
    // 10,000 samples from the distribution in question and computed the stats from that
    // sample to get min, 25%-ile, median and so on.  I did this 1000 times to get 5% and
    // 95% confidence limits for those values.

    // symmetrical, well behaved
    System.out.printf("normal\n");
    check(normal(10000),
      -4.417246, -3.419809,
      -0.6972919, -0.6519899,
      -0.02056658, 0.02176474,
      0.6503866, 0.6983311,
      4.419809, 5.417246,
      -0.01515753, 0.01592942,
      0.988395, 1.011883);

    // asymmetrical, well behaved.  The range for the maximum was fudged slightly to all this to pass.
    System.out.printf("exp\n");
    check(exp(10000),
            -3e-4, 3.278763e-04,
            0.2783866, 0.298,
            0.6765024, 0.7109463,
            1.356929, 1.414761,
            8, 20,
            0.983805, 1.015920,
            0.977162, 1.022093
    );

    // asymmetrical, wacko distribution where mean/median is about 200
    System.out.printf("gamma\n");
    check(gamma(10000, 0.1),
      -5e-30, 5e-30,                                    // minimum
      3.8e-6, 8.6e-6,                                   // 25th %-ile
      0.004847959, 0.007234259,                         // median
      0.3074556, 0.4049404,                             // 75th %-ile
      45, 100,                                          // maximum
      0.9, 1.1,                                         // mean
      2.8, 3.5);                                        // standard dev

  }

  private static void check(OnlineSummarizer x, double... values) {
    for (int i = 0; i < 5; i++) {
      checkRange("quartile " + i, x.getQuartile(i), values[2 * i], values[2 * i + 1]);
    }
    assertEquals(x.getQuartile(2), x.getMedian(), 0);

    checkRange("mean", x.getMean(), values[10], values[11]);
    checkRange("sd", x.getSD(), values[12], values[13]);
  }

  private static void checkRange(String msg, double v, double low, double high) {
    if (v < low || v > high) {
      fail("Wanted " + msg + " to be in range [" + low + ',' + high + "] but got " + v);
    }
  }

  private static OnlineSummarizer normal(int n) {
    OnlineSummarizer x = new OnlineSummarizer();
    Random gen = RandomUtils.getRandom(1L);
    for (int i = 0; i < n; i++) {
      x.add(gen.nextGaussian());
    }
    return x;
  }

  private static OnlineSummarizer exp(int n) {
    OnlineSummarizer x = new OnlineSummarizer();
    Random gen = RandomUtils.getRandom(1L);
    for (int i = 0; i < n; i++) {
      x.add(-Math.log1p(-gen.nextDouble()));
    }
    return x;
  }

  private static OnlineSummarizer gamma(int n, double shape) {
    OnlineSummarizer x = new OnlineSummarizer();
    Random gen = RandomUtils.getRandom();
    AbstractContinousDistribution gamma = new Gamma(shape, shape, gen);
    for (int i = 0; i < n; i++) {
      x.add(gamma.nextDouble());
    }
    return x;
  }

}


