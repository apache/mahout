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
import org.junit.Test;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public final class EvolutionaryProcessTest extends MahoutTestCase {

  @Test
  public void testConverges() throws Exception {
    State<Foo, Double> s0 = new State<Foo, Double>(new double[5], 1);
    s0.setPayload(new Foo());
    EvolutionaryProcess<Foo, Double> ep = new EvolutionaryProcess<Foo, Double>(10, 100, s0);

    State<Foo, Double> best = null;
    for (int i = 0; i < 20; i++) {
      best = ep.parallelDo(new EvolutionaryProcess.Function<Payload<Double>>() {
        @Override
        public double apply(Payload<Double> payload, double[] params) {
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

    ep.close();
    assertNotNull(best);
    assertEquals(0.0, best.getValue(), 0.02);
  }

  private static class Foo implements Payload<Double> {
    @Override
    public Foo copy() {
      return this;
    }

    @Override
    public void update(double[] params) {
      // ignore
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
      // no-op
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
      // no-op
    }
  }
}
