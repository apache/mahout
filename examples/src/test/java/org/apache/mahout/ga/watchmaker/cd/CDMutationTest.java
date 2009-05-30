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

import junit.framework.TestCase;
import org.apache.mahout.ga.watchmaker.cd.utils.MockDataSet;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.util.Random;

public class CDMutationTest extends TestCase {

  private Random rng;

  private MockDataSet mock;

  @Override
  protected void setUp() {
    rng = new MersenneTwisterRNG();
    mock = new MockDataSet(rng, 100);
  }

  /**
   * Test method for
   * {@link org.apache.mahout.ga.watchmaker.cd.CDMutation#rndDouble(double, double, double, java.util.Random)}.
   */
  public void testMutate() {
    DataSet dataset = DataSet.getDataSet();
    boolean modified = false; // true if at least one attribute has mutated

    int n = 100;
    for (int nloop = 0; nloop < n; nloop++) {
      mock.randomDataset();

      double range = rng.nextDouble();
      int k = rng.nextInt(1000);
      CDMutation mutation = new CDMutation(1.0, range, k);
      CDRule rule = new CDRule(0.0, rng);

      CDRule mutated = mutation.mutate(new CDRule(rule), rng);

      // check the ranges
      double min, max;
      double value, newval;
      int nbcats;

      for (int condInd = 0; condInd < mutated.getNbConditions(); condInd++) {
        int attrInd = CDRule.attributeIndex(condInd);
        value = rule.getV(condInd);
        newval = mutated.getV(condInd);
        modified = modified || (value != newval);

        if (dataset.isNumerical(attrInd)) {
          min = dataset.getMin(attrInd);
          max = dataset.getMax(attrInd);

          assertInRange(newval, min, max);
          assertTrue(Math.abs(newval - value) <= (max - min) * range);

        } else {
          nbcats = dataset.getNbValues(attrInd);

          assertInRange(newval, 0, nbcats);
        }
      }
      mock.verify();
    }
    
    assertTrue(modified);
  }

  private void assertInRange(double value, double min, double max) {
    TestCase.assertTrue("value < min", value >= min);
    TestCase.assertTrue("value > max", value <= max);
  }
}
