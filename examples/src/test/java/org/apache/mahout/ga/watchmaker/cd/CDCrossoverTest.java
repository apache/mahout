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

package org.apache.mahout.ga.watchmaker.cd;

import org.apache.mahout.examples.MahoutTestCase;
import org.easymock.EasyMock;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.List;
import java.util.Random;

public final class CDCrossoverTest extends MahoutTestCase {

  /**
   * if the parents have different values for all their genes, then the
   * offsprings will not any common gene.
   */
  @Test
  public void testMate1() {
    Random rng = RandomUtils.getRandom();

    // Initialize dataset
    DataSet dataset = EasyMock.createMock(DataSet.class);
    DataSet.initialize(dataset);

    int n = 100; // repeat this test n times
    int maxcrosspnts = 10;
    int maxattributes = 100;
    for (int nloop = 0; nloop < n; nloop++) {
      // we need at least 2 attributes for the crossover
      // and a label that will be skipped by the rules
      int nbattributes = rng.nextInt(maxattributes) + 3;
      int crosspnts = rng.nextInt(maxcrosspnts) + 1;

      // prepare dataset mock
      EasyMock.reset(dataset);
      EasyMock.expect(dataset.getNbAttributes()).andReturn(nbattributes).times(2);
      EasyMock.replay(dataset);

      CDCrossover crossover = new CDCrossover(crosspnts);

      // the parents have no gene in common
      CDRule parent0 = generate0Rule();
      CDRule parent1 = generate1Rule();

      List<CDRule> offsprings = crossover
          .mate(parent0, parent1, crosspnts, rng);
      assertEquals("offsprings number", 2, offsprings.size());
      CDRule offspring1 = offsprings.get(0);
      CDRule offspring2 = offsprings.get(1);

      // Check that the offspring have no gene in common
      for (int index = 0; index < offspring1.getNbConditions(); index++) {
        assertFalse("The offsprings have a common gene", CDRule.areGenesEqual(
            offspring1, offspring2, index));
      }
      
      EasyMock.verify(dataset);
    }
  }

  /**
   * Ensure that for a crossover of N points, the offsprings got N+1 different
   * areas.
   */
  @Test
  public void testMate2() {
    Random rng = RandomUtils.getRandom();

    // Initialize dataset
    DataSet dataset = EasyMock.createMock(DataSet.class);
    DataSet.initialize(dataset);

    int n = 100; // repeat this test n times
    int maxcrosspnts = 10;
    int maxattributes = 100;
    for (int nloop = 0; nloop < n; nloop++) {
      int nbattributes = rng.nextInt(maxattributes) + 3;
      int crosspnts = rng.nextInt(maxcrosspnts) + 1;
      // in the case of this test crosspnts should be < nbattributes
      if (crosspnts >= nbattributes) {
        crosspnts = nbattributes - 1;
      }

      // prepare dataset mock
      EasyMock.reset(dataset);
      EasyMock.expect(dataset.getNbAttributes()).andReturn(nbattributes).times(2);
      EasyMock.replay(dataset);

      CDCrossover crossover = new CDCrossover(crosspnts);

      // the parents have no gene in common
      CDRule parent0 = generate0Rule();
      CDRule parent1 = generate1Rule();

      // due to the random nature of the crossover their must be at most
      // (crosspnts+1) areas in the offsprings.
      int m = 10;

      for (int mloop = 0; mloop < m; mloop++) {
        List<CDRule> offsprings = crossover.mate(parent0, parent1, crosspnts,
            rng);
        assertEquals("offsprings number", 2, offsprings.size());

        // because the second offspring does not share any gene with the first
        // (see testMate1) we only need to verify one offspring
        CDRule offspring = offsprings.get(0);
        int nbareas = countAreas(offspring);
        assertTrue("NbAreas(" + nbareas + ") > crosspnts(" + crosspnts + ")+1",
            nbareas <= (crosspnts + 1));
      }

      EasyMock.verify(dataset);
    }

  }

  static int countAreas(CDRule rule) {

    int nbareas = 1; // we already start in an area
    int partind = 0; // index of the start of the current part

    for (int index = 0; index < rule.getNbConditions(); index++) {
      if (!rule.areGenesEqual(partind, index)) {
        // we are in a new area
        nbareas++;
        partind = index;
      }
    }

    return nbareas;
  }

  static CDRule generate0Rule() {
    CDRule rule = new CDRule(1);

    for (int index = 0; index < rule.getNbConditions(); index++) {
      rule.setW(index, 0);
      rule.setO(index, false);
      rule.setV(index, 0);
    }

    return rule;
  }

  static CDRule generate1Rule() {
    CDRule rule = new CDRule(1);

    for (int index = 0; index < rule.getNbConditions(); index++) {
      rule.setW(index, 1);
      rule.setO(index, true);
      rule.setV(index, 10);
    }

    return rule;
  }
}
