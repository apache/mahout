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

package org.apache.mahout.ep;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Random;

public final class ThreadedEvolutionaryProcessTest extends MahoutTestCase {
  @Test
  public void testOptimize() throws Exception {
    ThreadedEvolutionaryProcess ep = new ThreadedEvolutionaryProcess(50);
    final Random random = RandomUtils.getRandom();
    ep.optimize(new ThreadedEvolutionaryProcess.Function() {
      /**
       * Implements a skinny quadratic bowl.
       */
      @Override
      public double apply(double[] params) {
        double sum = 0;
        int i = 0;
        for (double x : params) {
          x = (i + 1) * (x - i);
          i++;
          sum += x * x;
        }
        try {
          // variable delays to emulate a tricky function
          Thread.sleep((long) Math.floor(-2 * Math.log(1 - random.nextDouble())));
        } catch (InterruptedException e) {
          // ignore interruptions
        }

        return -sum;
      }
    }, 5, 200, 2);

    System.out.println(ep);
    // disabled due to non-repeatability
    /*
    double[] r = x.getMappedParams();
    int i = 0;
    for (double v : r) {
      assertEquals(String.format(Locale.ENGLISH, "Coordinate %d", i), i, v, 0.02);
      i++;
    }
     */
  }
}
