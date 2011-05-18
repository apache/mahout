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
import org.apache.mahout.math.jet.random.Poisson;
import org.junit.Test;

import java.util.Random;

public class OnlineExponentialAverageTest extends MahoutTestCase {
  @Test
  public void testAverage() {
    double[] t = {11.35718, 21.54637, 28.91061, 33.03586, 39.57767};
    double[] x = {1.5992071, -1.3577032, -0.3405638, 0.7048632, 0.3020558};
    double[] m = {1.5992071, -1.0168100, -0.4797436, 0.2836447, 0.2966159};

    OnlineExponentialAverage averager = new OnlineExponentialAverage(5);

    for (int i = 0; i < t.length; i++) {
      averager.add(t[i], x[i]);
      assertEquals("Step " + i, m[i], averager.mean(), 1.0e-6);
    }
  }

  @Test
  public void testRate() {
    Random gen = RandomUtils.getRandom();

    Poisson p = new Poisson(5, gen);
    double lastT = 0;

    double[] k = new double[1000];
    double[] t = new double[1000];
    for (int i = 1; i < 1000; i++) {
      // we sample every 5-15 seconds
      double dt = gen.nextDouble() * 10 + 5;
      t[i] = lastT + dt;

      // at those points, we get a Poisson count
      k[i] = p.nextInt(dt * 0.2);
      lastT = t[i];
    }

    OnlineExponentialAverage averager = new OnlineExponentialAverage(2000);

    for (int i = 1; i < 1000; i++) {
      averager.add(t[i], k[i]);
    }

    assertEquals("Expected rate", 0.2, averager.meanRate(), 0.01);
  }
}
