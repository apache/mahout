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
import org.apache.mahout.ga.watchmaker.cd.utils.MockDataSet;
import org.apache.mahout.common.RandomUtils;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public final class CDRuleTest extends MahoutTestCase {

  private Random rng;
  private MockDataSet mock;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    rng = RandomUtils.getRandom();
    mock = new MockDataSet(rng, 50);
  }

  /**
   * Test method for
   * {@link org.apache.mahout.ga.watchmaker.cd.CDFactory#generateRandomCandidate(java.util.Random)}.
   */
  public void testRandomCDRule() {
    DataSet dataset = DataSet.getDataSet();
    double threshold = 0.0;

    int n = 100;
    for (int nloop = 0; nloop < n; nloop++) {
      mock.randomDataset();

      CDRule rule = new CDRule(threshold, rng);
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        int attrInd = CDRule.attributeIndex(condInd);
        
        CDMutationTest.assertInRange(rule.getW(condInd), 0, 1);
        
        if (dataset.isNumerical(attrInd)) {
          CDMutationTest.assertInRange(rule.getV(condInd), dataset.getMin(attrInd), dataset
              .getMax(attrInd));
        } else {
          CDMutationTest.assertInRange(rule.getV(condInd), 0, dataset.getNbValues(attrInd) - 1);
        }
      }

      mock.verify();
    }
  }

  /**
   * Test the Weight part of the condition.
   * 
   */
  @Test
  public void testWCondition() {

    // the dataline has all its attributes set to 0d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribute(EasyMock.anyInt())).andReturn(0.0).atLeastOnce();
    EasyMock.replay(dl);

    // all the conditions are : attribut < 0
    int n = 100; // repeat the test n times
    for (int nloop = 0; nloop < n; nloop++) {
      double thr = rng.nextDouble();

      mock.numericalDataset();

      CDRule rule = new CDRule(thr);
      for (int index = 0; index < rule.getNbConditions(); index++) {
        rule.setW(index, rng.nextDouble());
        rule.setO(index, false);
        rule.setV(index, 0);
      }

      // all coditions should return false unless w < threshold
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        assertEquals(rule.getW(condInd) < thr, rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

  /**
   * Test the Operator part of the condition, on numerical attributes
   * 
   */
  @Test
  public void testOConditionNumerical() {

    // the dataline has all its attributes set to 1d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribute(EasyMock.anyInt())).andReturn(1.0).atLeastOnce();
    EasyMock.replay(dl);

    int n = 100; // repeat the test n times
    for (int nloop = 0; nloop < n; nloop++) {
      mock.numericalDataset();

      CDRule rule = new CDRule(0.0);
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        rule.setW(condInd, 1.0); // all weights are 1 (active)
        rule.setO(condInd, rng.nextBoolean());
        rule.setV(condInd, 0);
      }

      // the condition is true if the operator is >=
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        assertEquals(rule.getO(condInd), rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

  /**
   * Test the Operator part of the condition, on numerical attributes
   * 
   */
  @Test
  public void testOConditionCategorical() {

    // the dataline has all its attributes set to 1d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribute(EasyMock.anyInt())).andReturn(1.0).atLeastOnce();
    EasyMock.replay(dl);

    int n = 100; // repeat the test n times
    for (int nloop = 0; nloop < n; nloop++) {
      mock.categoricalDataset();

      // all weights are 1 (active)
      CDRule rule = new CDRule(0.0);
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        rule.setW(condInd, 1.0);
        rule.setO(condInd, rng.nextBoolean());
        rule.setV(condInd, rng.nextInt(2)); // two categories
      }

      // the condition is true if the operator is == and the values are equal
      // (value==1), or the operator is != and the values are no equal
      // (value==0)
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        assertEquals(!(rule.getO(condInd) ^ rule.getV(condInd) == 1.0),
                     rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

}
