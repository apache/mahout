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

package org.apache.mahout.math;

import com.google.common.base.Splitter;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Random;

public final class TestRandomAccessSparseVector extends AbstractVectorTest<RandomAccessSparseVector> {

  @Override
  Vector generateTestVector(int cardinality) {
    return new RandomAccessSparseVector(cardinality);
  }


  @Override
  public RandomAccessSparseVector vectorToTest(int size) {
    RandomAccessSparseVector r = new RandomAccessSparseVector(size);
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < 3; i++) {
      r.set(gen.nextInt(r.size()), gen.nextGaussian());
    }
    return r;
  }

  @Override
  @Test
  public void testToString() {
    Vector w;
    w = generateTestVector(20);
    w.set(0, 1.1);
    w.set(13, 100500.);
    w.set(19, 3.141592);

    for (String token : Splitter.on(',').split(w.toString().substring(1, w.toString().length() - 2))) {
      String[] tokens = token.split(":");
      assertEquals(Double.parseDouble(tokens[1]), w.get(Integer.parseInt(tokens[0])), 0.0);
    }

    w = generateTestVector(12);
    w.set(10, 0.1);
    assertEquals("{10:0.1}", w.toString());

    w = generateTestVector(12);
    assertEquals("{}", w.toString());
  }
}
