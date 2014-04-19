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

package org.apache.mahout.classifier.evaluation;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.jet.random.Normal;
import org.junit.Test;

import java.util.Random;

public final class AucTest extends MahoutTestCase {

  @Test
  public void testAuc() {
    Auc auc = new Auc();
    Random gen = RandomUtils.getRandom();
    auc.setProbabilityScore(false);
    for (int i=0; i<100000; i++) {
      auc.add(0, gen.nextGaussian());
      auc.add(1, gen.nextGaussian() + 1);
    }
    assertEquals(0.76, auc.auc(), 0.01);
  }

  @Test
  public void testTies() {
    Auc auc = new Auc();
    Random gen = RandomUtils.getRandom();
    auc.setProbabilityScore(false);
    for (int i=0; i<100000; i++) {
      auc.add(0, gen.nextGaussian());
      auc.add(1, gen.nextGaussian() + 1);
    }

    // ties outside the normal range could cause index out of range
    auc.add(0, 5.0);
    auc.add(0, 5.0);
    auc.add(0, 5.0);
    auc.add(0, 5.0);

    auc.add(1, 5.0);
    auc.add(1, 5.0);
    auc.add(1, 5.0);

    assertEquals(0.76, auc.auc(), 0.05);
  }

  @Test
  public void testEntropy() {
    Auc auc = new Auc();
    Random gen = RandomUtils.getRandom();
    Normal n0 = new Normal(-1, 1, gen);
    Normal n1 = new Normal(1, 1, gen);
    for (int i=0; i<100000; i++) {
      double score = n0.nextDouble();
      double p = n1.pdf(score) / (n0.pdf(score) + n1.pdf(score));
      auc.add(0, p);

      score = n1.nextDouble();
      p = n1.pdf(score) / (n0.pdf(score) + n1.pdf(score));
      auc.add(1, p);
    }
    Matrix m = auc.entropy();
    assertEquals(-0.35, m.get(0, 0), 0.02);
    assertEquals(-2.36, m.get(0, 1), 0.02);
    assertEquals(-2.36, m.get(1, 0), 0.02);
    assertEquals(-0.35, m.get(1, 1), 0.02);
  }
}
