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

import org.apache.mahout.common.RandomUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ExecutionException;

public class EvolutionaryProcessTest {

  @Before
  public void setUp() {
    RandomUtils.useTestSeed();
  }

  @Test
  public void converges() throws ExecutionException, InterruptedException {
    State<Foo> s0 = new State<Foo>(new double[5], 1);
    s0.setPayload(new Foo());
    s0.setRand(new Random(1));
    EvolutionaryProcess<Foo> ep = new EvolutionaryProcess<Foo>(10, 100, s0);

    State<Foo> best = null;
    for (int i = 0; i < 20  ; i++) {
      best = ep.parallelDo(new EvolutionaryProcess.Function<Foo>() {
        @Override
        public double apply(Foo payload, double[] params) {
          int i = 1;
          double sum = 0;
          for (double x : params) {
            sum += i * (x - i) * (x - i);
            i++;
          }
          return -sum;
        }
      });

      ep.mutatePopulation(3);

      System.out.printf("%10.3f %.3f\n", best.getValue(), best.getOmni());
    }

    Assert.assertNotNull(best);
    Assert.assertEquals(0, best.getValue(), 0.02);
  }

  private static class Foo implements Payload<Foo> {
    @Override
    public Foo copy() {
      return this;
    }

    @Override
    public void update(double[] params) {
      // ignore
    }
  }
}
