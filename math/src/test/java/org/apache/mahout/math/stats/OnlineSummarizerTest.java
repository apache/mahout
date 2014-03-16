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

import java.util.Arrays;
import java.util.Random;

public final class OnlineSummarizerTest extends MahoutTestCase {

  @Test
  public void testStats() {
   /**
     the reference limits here were derived using a numerical simulation where I took
     10,000 samples from the distribution in question and computed the stats from that
     sample to get min, 25%-ile, median and so on. I did this 1000 times to get 5% and
     95% confidence limits for those values.
   */

    //symmetrical, well behaved
    System.out.printf("normal\n");
    check(normal(10000));

    //asymmetrical, well behaved. The range for the maximum was fudged slightly to all this to pass.
    System.out.printf("exp\n");
    check(exp(10000));

    //asymmetrical, wacko distribution where mean/median is about 200
    System.out.printf("gamma\n");
    check(gamma(10000, 0.1));
  }

  private static void check(double[] samples) {
    OnlineSummarizer s = new OnlineSummarizer();
    double mean = 0;
    double sd = 0;
    int n = 1;
    for (double x : samples) {
      s.add(x);
      double old = mean;
      mean += (x - mean) / n;
      sd += (x - old) * (x - mean);
      n++;
    }
    sd = Math.sqrt(sd / samples.length);

    Arrays.sort(samples);

    for (int i = 0; i < 5; i++) {
      int index = Math.abs(Arrays.binarySearch(samples, s.getQuartile(i)));
      assertEquals("quartile " + i, i * (samples.length - 1) / 4.0, index, 10);
    }
    assertEquals(s.getQuartile(2), s.getMedian(), 0);

    assertEquals("mean", s.getMean(), mean, 0);
    assertEquals("sd", s.getSD(), sd, 1e-8);
  }

  private static double[] normal(int n) {
    double[] r = new double[n];
    Random gen = RandomUtils.getRandom(1L);
    for (int i = 0; i < n; i++) {
      r[i] = gen.nextGaussian();
    }
    return r;
  }

  private static double[] exp(int n) {
    double[] r = new double[n];
    Random gen = RandomUtils.getRandom(1L);
    for (int i = 0; i < n; i++) {
      r[i] = -Math.log1p(-gen.nextDouble());
    }
    return r;
  }

  private static double[] gamma(int n, double shape) {
    double[] r = new double[n];
    Random gen = RandomUtils.getRandom();
    AbstractContinousDistribution gamma = new Gamma(shape, shape, gen);
    for (int i = 0; i < n; i++) {
      r[i] = gamma.nextDouble();
    }
    return r;
  }
}


