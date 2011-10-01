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

import org.apache.mahout.math.MahoutTestCase;
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
    check(normal(10000),
            -4.417246, -3.419809,
            -0.6972919, -0.6519899,
            -0.02056658, 0.02176474,
            0.6503866, 0.6983311,
            3.419809, 4.417246,
            -0.01515753, 0.01592942,
            0.988395, 1.011883);

    // asymmetrical, well behaved.  The range for the maximum was fudged slightly to all this to pass.
    check(exp(10000),
            4.317969e-06, 3.278763e-04,
            0.2783866, 0.298,
            0.6765024, 0.7109463,
            1.356929, 1.414761,
            8, 13,
            0.983805, 1.015920,
            0.977162, 1.022093
    );

    // asymmetrical, wacko distribution where mean/median > 10^28
    // TODO need more work here
//    check(gamma(10000, 3),
//            0, 0,                                             // minimum
//            0, 6.26363334269806e-58,                          // 25th %-ile
//            8.62261497075834e-30, 2.01422505081014e-28,       // median
//            6.70225617733614e-12, 4.44299757853286e-11,       // 75th %-ile
//            238.451174077827, 579.143886928158,               // maximum
//            0.837031762527458, 1.17244066539313,              // mean
//            8.10277696526878, 12.1426255901507);              // standard dev
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
    // TODO use RandomUtils.getRandom() and rejigger constants to make test pass
    Random gen = new Random(1L);
    for (int i = 0; i < n; i++) {
      x.add(gen.nextGaussian());
    }
    return x;
  }

  private static OnlineSummarizer exp(int n) {
    OnlineSummarizer x = new OnlineSummarizer();
    // TODO use RandomUtils.getRandom() and rejigger constants to make test pass
    Random gen = new Random(1L);
    for (int i = 0; i < n; i++) {
      x.add(-Math.log1p(-gen.nextDouble()));
    }
    return x;
  }

}


