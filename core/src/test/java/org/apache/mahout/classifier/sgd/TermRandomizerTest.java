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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class TermRandomizerTest {
  List<String> wordList;

  @Before
  public void setup() throws IOException {
    wordList = Resources.readLines(Resources.getResource("word-list.txt"), Charsets.UTF_8);
  }

  @Test
  public void testBinaryRandomizer1() {
    TermRandomizer r = new BinaryRandomizer(1, 100);
    Vector sum = new DenseVector(100);
    for (String s : wordList) {
      Vector v = r.randomizedInstance(Arrays.asList(s), 0, false);
      sum.assign(v, Functions.plus);

      // should have exactly one value set
      Assert.assertEquals(1, v.zSum(), 0);
      Assert.assertEquals(1, v.maxValue(), 0);
    }

    Assert.assertTrue(sum.maxValue() < 11);

    int total = 0;
    int[] counts = new int[11];
    for (int i = 0; i < 100; i++) {
      counts[(int) sum.get(i)]++;
      total += sum.get(i);
    }

    Assert.assertEquals(512, total);

    int max = 0;
    int maxCount = counts[0];
    for (int i = 0; i < counts.length; i++) {
      int count = counts[i];
      if (count > maxCount) {
        max = i;
        maxCount = count;
      }
    }

    // 500 words across 100 slots should have a peak at 5 words
    Assert.assertEquals(5, max);

    // regression test... should be very stable if has functions are not changed
    Assert.assertArrayEquals(new int[]{2, 2, 3, 12, 20, 25, 14, 9, 5, 4, 4}, counts);
  }

  @Test
  public void testBinaryRandomizer2() {
    TermRandomizer r = new BinaryRandomizer(3, 100);
    Vector sum = new DenseVector(100);

    int collisions = 0;
    for (String s : wordList) {
      Vector v = r.randomizedInstance(Arrays.asList(s), 0, false);
      sum.assign(v, Functions.plus);

      // should have exactly three values set
      Assert.assertEquals(3, v.zSum(), 0);

      Assert.assertTrue(v.maxValue() < 3);
      if (v.maxValue() == 2) {
        collisions++;
      }
    }
    // regression test.  The real desire is to know that there aren't too many collisions
    Assert.assertEquals(11, collisions);

    // 500 words across 100 slots should have a peak at 5 words
    Assert.assertEquals(25, sum.maxValue(), 0);

    int total = 0;
    int[] counts = new int[26];
    for (int i = 0; i < 100; i++) {
      counts[(int) sum.get(i)]++;
      total += sum.get(i);
    }

    Assert.assertEquals(3 * 512, total);

    int max = 0;
    int maxCount = counts[0];
    for (int i = 0; i < counts.length; i++) {
      int count = counts[i];
      if (count > maxCount) {
        max = i;
        maxCount = count;
      }
    }

    // regression test... The exact position of this max is not super deterministic because of the broad peak.
    Assert.assertEquals(13, max);

    // regression test... should be very stable if has functions are not changed
    Assert.assertArrayEquals(new int[]{0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 8, 9, 13, 8, 8, 13, 13, 6, 2, 4, 4, 3, 2, 1, 1}, counts);
  }

  @Test
  public void testDenseRandomizer() {
    TermRandomizer r = new DenseRandomizer(100);
    Vector sum = new DenseVector(100);

    Vector prev = null;
    Vector dotDistribution = new DenseVector(511);
    int i = 0;
    for (String s : wordList) {
      Vector v = r.randomizedInstance(Arrays.asList(s), 0, false);
      sum.assign(v, Functions.plus);

      if (prev != null) {
        dotDistribution.set(i, v.dot(prev));
        i++;
      }

      // mean should be close to zero
      Assert.assertEquals(0, v.zSum() / 100, 1);
      // standard deviation should be near 1
      Assert.assertEquals(1, Math.sqrt(v.dot(v)/99), 0.3);
      prev = v;
    }

    // dot products of unrelated vectors should be near zero
    Assert.assertEquals(0, dotDistribution.zSum() / 511, 1);
    // standard deviation should be about sqrt(100)
    Assert.assertEquals(10, Math.sqrt(dotDistribution.dot(dotDistribution) / 510), 2);
  }
}
