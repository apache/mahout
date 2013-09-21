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

package org.apache.mahout.math.set;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import junit.framework.TestCase;
import org.apache.mahout.common.RandomUtils;

import java.util.Collection;
import java.util.List;
import java.util.Random;

public class HashUtilsTest extends TestCase {
  public void testHashFloat() {
    Multiset<Integer> violations = HashMultiset.create();
    for (int k = 0; k < 1000; k++) {
      List<Float> original = Lists.newArrayList();

      Random gen = RandomUtils.getRandom();
      for (int i = 0; i < 10000; i++) {
        float x = (float) gen.nextDouble();
        original.add(x);
      }

      violations.add(checkCounts(original) <= 12 ? 0 : 1);
    }
    // the hashes for floats don't really have 32 bits of entropy so the test
    // only succeeds at better than about 99% rate.
    assertTrue(violations.count(0) >= 985);
  }

  public void testHashDouble() {
    List<Double> original = Lists.newArrayList();

    for (int k = 0; k < 10; k++) {
      Random gen = RandomUtils.getRandom();
      for (int i = 0; i < 10000; i++) {
        double x = gen.nextDouble();
        original.add(x);
      }

      checkCounts(original);
    }
  }

  public void testHashLong() {
    List<Long> original = Lists.newArrayList();

    for (int k = 0; k < 10; k++) {
      Random gen = RandomUtils.getRandom();
      for (int i = 0; i < 10000; i++) {
        long x = gen.nextLong();
        original.add(x);
      }

      checkCounts(original);
    }
  }

  private static <T> int checkCounts(Collection<T> original) {
    Multiset<T> hashCounts = HashMultiset.create();
    for (T v : original) {
      hashCounts.add(v);
    }

    Multiset<Integer> countCounts = HashMultiset.create();
    for (T hash : hashCounts) {
      countCounts.add(hashCounts.count(hash));
    }

    return original.size() - countCounts.count(1);
  }
}
