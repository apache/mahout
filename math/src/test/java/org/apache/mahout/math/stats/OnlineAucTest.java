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

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class OnlineAucTest {
  @Test
  public void testBinaryCase() {
    OnlineAuc a1 = new OnlineAuc();
    a1.setRandom(new Random(1));
    a1.setPolicy(OnlineAuc.ReplacementPolicy.FAIR);

    OnlineAuc a2 = new OnlineAuc();
    a2.setRandom(new Random(2));
    a2.setPolicy(OnlineAuc.ReplacementPolicy.FIFO);

    OnlineAuc a3 = new OnlineAuc();
    a3.setRandom(new Random(3));
    a3.setPolicy(OnlineAuc.ReplacementPolicy.RANDOM);

    Random gen = new Random(1);
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
