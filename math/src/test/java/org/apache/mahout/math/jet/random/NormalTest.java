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

package org.apache.mahout.math.jet.random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.MahoutTestCase;
import org.junit.Test;

import java.util.Random;

public final class NormalTest extends MahoutTestCase {

  private final double[] breaks =
      {-1.2815516, -0.8416212, -0.5244005, -0.2533471, 0.0000000, 0.2533471, 0.5244005, 0.8416212, 1.2815516};
  private final double[] quantiles = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  @Test
  public void testCdf() {
    Random gen = RandomUtils.getRandom();
    double offset = 0;
    double scale = 1;
    for (int k = 0; k < 20; k++) {
      Normal dist = new Normal(offset, scale, null);
      DistributionChecks.checkCdf(offset, scale, dist, breaks, quantiles);
      offset = gen.nextGaussian();
      scale = Math.exp(3 * gen.nextGaussian());
    }
  }

  @Test
  public void consistency() {
    Random gen = RandomUtils.getRandom();
    double offset = 0;
    double scale = 1;
    Normal dist = new Normal(offset, scale, RandomUtils.getRandom());
    for (int k = 0; k < 20; k++) {
      dist.setState(offset, scale);
      DistributionChecks.checkDistribution(dist, breaks, offset, scale, 10000);
      offset = gen.nextGaussian();
      scale = Math.exp(3 * gen.nextGaussian());
    }
  }

  @Test
  public void testSetState() throws Exception {
    Normal dist = new Normal(0, 1, RandomUtils.getRandom());
    dist.setState(1.3, 5.9);
    DistributionChecks.checkDistribution(dist, breaks, 1.3, 5.9, 10000);
  }

  @Test
  public void testToString() {
    assertEquals("org.apache.mahout.math.jet.random.Normal(m=1.300000, sd=5.900000)",
        new Normal(1.3, 5.9, null).toString());
  }
}
