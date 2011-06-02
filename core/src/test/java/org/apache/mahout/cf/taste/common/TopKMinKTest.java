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

package org.apache.mahout.cf.taste.common;

import org.junit.Test;

import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * tests for {@link TopK} and {@link MinK}
 */
public class TopKMinKTest {

  static class IntComparator implements Comparator<Integer> {
    @Override
    public int compare(Integer one, Integer two) {
      return one.compareTo(two);
    }
  }

  @Test
  public void oneMillionTop() {
    TopK<Integer> topFiveNumbers = new TopK<Integer>(5, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      topFiveNumbers.offer(n);
    }

    List<Integer> numbers = topFiveNumbers.retrieve();
    assertEquals(new Integer(1000000), numbers.get(0));
    assertEquals(new Integer(999999), numbers.get(1));
    assertEquals(new Integer(999998), numbers.get(2));
    assertEquals(new Integer(999997), numbers.get(3));
    assertEquals(new Integer(999996), numbers.get(4));
  }

  @Test
  public void oneMillionSmallestTop() {
    TopK<Integer> topFiveNumbers = new TopK<Integer>(5, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      topFiveNumbers.offer(n);
    }

    assertEquals(new Integer(999996), topFiveNumbers.smallestGreat());
  }

  @Test
  public void oneMillionMin() {
    MinK<Integer> minFiveNumbers = new MinK<Integer>(5, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      minFiveNumbers.offer(n);
    }

    List<Integer> numbers = minFiveNumbers.retrieve();
    assertEquals(new Integer(1), numbers.get(0));
    assertEquals(new Integer(2), numbers.get(1));
    assertEquals(new Integer(3), numbers.get(2));
    assertEquals(new Integer(4), numbers.get(3));
    assertEquals(new Integer(5), numbers.get(4));
  }

  @Test
  public void oneMillionGreatestMin() {
    MinK<Integer> minFiveNumbers = new MinK<Integer>(5, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      minFiveNumbers.offer(n);
    }

    assertEquals(new Integer(5), minFiveNumbers.greatestSmall());
  }

  @Test
  public void oneMillionTwoTimesTop() {
    TopK<Integer> topThreeNumbers = new TopK<Integer>(3, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      topThreeNumbers.offer(n);
      topThreeNumbers.offer(n);
    }

    List<Integer> numbers = topThreeNumbers.retrieve();
    assertEquals(new Integer(1000000), numbers.get(0));
    assertEquals(new Integer(1000000), numbers.get(1));
    assertEquals(new Integer(999999), numbers.get(2));
  }

  @Test
  public void oneMillionTwoTimesMin() {
    MinK<Integer> minThreeNumbers = new MinK<Integer>(3, new IntComparator());
    for (int n = 1; n <= 1000000; n++) {
      minThreeNumbers.offer(n);
      minThreeNumbers.offer(n);
    }

    List<Integer> numbers = minThreeNumbers.retrieve();
    assertEquals(new Integer(1), numbers.get(0));
    assertEquals(new Integer(1), numbers.get(1));
    assertEquals(new Integer(2), numbers.get(2));
  }
}