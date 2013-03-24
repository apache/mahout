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

package org.apache.mahout.fpm.pfpgrowth.fpgrowth;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import com.google.common.collect.Sets;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

public final class FrequentPatternMaxHeapTest extends MahoutTestCase {

  @Test
  public void testMapHeap() {
    Random gen = RandomUtils.getRandom();

    FrequentPatternMaxHeap pq = new FrequentPatternMaxHeap(50, true);
    for (int i = 0; i < 20; i++) {
      FrequentPatternMaxHeap rs = new FrequentPatternMaxHeap(50, false);
      for (int j = 0; j < 1000; j++) {
        Pattern p = generateRandomPattern(gen);
        rs.insert(p);
      }
      for (Pattern p : rs.getHeap()) {
        pq.insert(p);
      }
    }
  }

  private static Pattern generateRandomPattern(Random gen) {
    int length = 1 + Math.abs(gen.nextInt() % 6);
    Pattern p = new Pattern();
    Collection<Integer> set = Sets.newHashSet();
    for (int i = 0; i < length; i++) {
      int id = Math.abs(gen.nextInt() % 20);
      while (set.contains(id)) {
        id = Math.abs(gen.nextInt() % 20);
      }
      set.add(id);
      int s = 5 + gen.nextInt() % 4;
      p.add(id, s);
    }
    Arrays.sort(p.getPattern());
    return p;
  }
}
