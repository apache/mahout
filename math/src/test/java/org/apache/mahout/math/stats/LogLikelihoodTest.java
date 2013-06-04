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

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.util.List;
import java.util.Random;

public final class LogLikelihoodTest extends MahoutTestCase {

  @Test
  public void testEntropy() throws Exception {

    assertEquals(1.386294, LogLikelihood.entropy(1, 1), 0.0001);
    assertEquals(0.0, LogLikelihood.entropy(1), 0.0);
    //TODO: more tests here
    try {
      LogLikelihood.entropy(-1, -1); //exception
      fail();
    } catch (IllegalArgumentException e) {

    }
  }

  @Test
  public void testLogLikelihood() throws Exception {
    //TODO: check the epsilons
    assertEquals(2.772589, LogLikelihood.logLikelihoodRatio(1, 0, 0, 1), 0.000001);
    assertEquals(27.72589, LogLikelihood.logLikelihoodRatio(10, 0, 0, 10), 0.00001);
    assertEquals(39.33052, LogLikelihood.logLikelihoodRatio(5, 1995, 0, 100000), 0.00001);
    assertEquals(4730.737, LogLikelihood.logLikelihoodRatio(1000, 1995, 1000, 100000), 0.001);
    assertEquals(5734.343, LogLikelihood.logLikelihoodRatio(1000, 1000, 1000, 100000), 0.001);
    assertEquals(5714.932, LogLikelihood.logLikelihoodRatio(1000, 1000, 1000, 99000), 0.001);
  }

  @Test
  public void testRootLogLikelihood() {
    // positive where k11 is bigger than expected.
    assertTrue(LogLikelihood.rootLogLikelihoodRatio(904, 21060, 1144, 283012) > 0.0);

    // negative because k11 is lower than expected
    assertTrue(LogLikelihood.rootLogLikelihoodRatio(36, 21928, 60280, 623876) < 0.0);

    assertEquals(Math.sqrt(2.772589), LogLikelihood.rootLogLikelihoodRatio(1, 0, 0, 1), 0.000001);
    assertEquals(-Math.sqrt(2.772589), LogLikelihood.rootLogLikelihoodRatio(0, 1, 1, 0), 0.000001);
    assertEquals(Math.sqrt(27.72589), LogLikelihood.rootLogLikelihoodRatio(10, 0, 0, 10), 0.00001);

    assertEquals(Math.sqrt(39.33052), LogLikelihood.rootLogLikelihoodRatio(5, 1995, 0, 100000), 0.00001);
    assertEquals(-Math.sqrt(39.33052), LogLikelihood.rootLogLikelihoodRatio(0, 100000, 5, 1995), 0.00001);

    assertEquals(Math.sqrt(4730.737), LogLikelihood.rootLogLikelihoodRatio(1000, 1995, 1000, 100000), 0.001);
    assertEquals(-Math.sqrt(4730.737), LogLikelihood.rootLogLikelihoodRatio(1000, 100000, 1000, 1995), 0.001);

    assertEquals(Math.sqrt(5734.343), LogLikelihood.rootLogLikelihoodRatio(1000, 1000, 1000, 100000), 0.001);
    assertEquals(Math.sqrt(5714.932), LogLikelihood.rootLogLikelihoodRatio(1000, 1000, 1000, 99000), 0.001);
  }

  @Test
  public void testRootNegativeLLR() {
    assertTrue(LogLikelihood.rootLogLikelihoodRatio(6, 7567, 1924, 2426487) >= 0.0);
  }

  @Test
  public void testFrequencyComparison() {
    final Random rand = RandomUtils.getRandom();

    // build a vector full of sample from exponential distribuiton
    // this will have lots of little positive values and a few big ones
    Vector p1 = new DenseVector(25)
      .assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return -Math.log1p(-rand.nextDouble());
        }
      });

    // make a copy
    Vector p2 = p1.like().assign(p1);

    // nuke elements 0..4
    p1.viewPart(0, 5).assign(0);

    // and boost elements 5..7
    p1.viewPart(5, 3).assign(Functions.mult(4));

    // then normalize to turn it into a probability distribution
    p1.assign(Functions.div(p1.norm(1)));

    // likewise normalize p2
    p2.assign(Functions.div(p2.norm(1)));

    // sample 100 times from p1
    Multiset<Integer> w1 = HashMultiset.create();
    for (int i = 0; i < 100; i++) {
      w1.add(sample(p1, rand));
    }

    // and 1000 times from p2
    Multiset<Integer> w2 = HashMultiset.create();
    for (int i = 0; i < 1000; i++) {
      w2.add(sample(p2, rand));
    }

    // comparing frequencies, we should be able to find 8 items with score > 0
    List<LogLikelihood.ScoredItem<Integer>> r = LogLikelihood.compareFrequencies(w1, w2, 8, 0);
    assertTrue(r.size() <= 8);
    assertFalse(r.isEmpty());
    for (LogLikelihood.ScoredItem<Integer> item : r) {
      assertTrue(item.getScore() >= 0);
    }

    // the most impressive should be 7
    assertEquals(7, (int) r.get(0).getItem());

    // make sure scores are descending
    double lastScore = r.get(0).getScore();
    for (LogLikelihood.ScoredItem<Integer> item : r) {
      assertTrue(item.getScore() <= lastScore);
      lastScore = item.getScore();
    }

    // now as many as have score >= 1
    r = LogLikelihood.compareFrequencies(w1, w2, 40, 1);

    // only the boosted items should make the cut
    assertEquals(2, r.size());
    assertEquals(7, (int) r.get(0).getItem());
    assertEquals(6, (int) r.get(1).getItem());

    r = LogLikelihood.compareFrequencies(w1, w2, 1000, -100);
    Multiset<Integer> k = HashMultiset.create();
    for (LogLikelihood.ScoredItem<Integer> item : r) {
      k.add(item.getItem());
    }
    for (int i = 0; i < 25; i++) {
      assertTrue("i = " + i, k.count(i) == 1 || w2.count(i) == 0);
    }

    // all values that had non-zero counts in larger set should have result scores
    assertEquals(w2.elementSet().size(), r.size());
    assertEquals(7, (int) r.get(0).getItem());
    assertEquals(6, (int) r.get(1).getItem());
    
    // the last item should definitely have negative score
    assertTrue(r.get(r.size() - 1).getScore() < 0);

    // make sure scores are descending
    lastScore = r.get(0).getScore();
    for (LogLikelihood.ScoredItem<Integer> item : r) {
      assertTrue(item.getScore() <= lastScore);
      lastScore = item.getScore();
    }
  }

  /**
   * Samples from a multinomial distribution with parameters p and random generator rand.
   * @param p      A vector describing the distribution.  Should sum to 1.
   * @param rand   A random number generator.
   * @return  A single sample from the multinomial distribution.
   */
  private static int sample(Vector p, Random rand) {
    double u = rand.nextDouble();

    // simple sequential algorithm.  Not the fastest, but we don't care
    for (int i = 0; i < p.size(); i++) {
      if (u <= p.get(i)) {
        return i;
      }
      u -= p.get(i);
    }
    return p.size() - 1;
  }
}
