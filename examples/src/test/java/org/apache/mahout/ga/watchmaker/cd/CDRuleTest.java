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
import junit.framework.Assert;
import org.apache.mahout.ga.watchmaker.cd.utils.MockDataSet;
import org.uncommons.maths.random.MersenneTwisterRNG;
import org.easymock.classextension.EasyMock;

import java.util.Random;

public class CDRuleTest extends TestCase {

  private Random rng;

  private MockDataSet mock;

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
        
        assertInRange(rule.getW(condInd), 0, 1);
        
        if (dataset.isNumerical(attrInd)) {
          assertInRange(rule.getV(condInd), dataset.getMin(attrInd), dataset
              .getMax(attrInd));
        } else {
          assertInRange(rule.getV(condInd), 0, dataset.getNbValues(attrInd) - 1);
        }
      }

      mock.verify();
    }
  }

  private void assertInRange(double value, double min, double max) {
    Assert.assertTrue("value < min", value >= min);
    Assert.assertTrue("value > max", value <= max);
  }

  @Override
  protected void setUp() {
    rng = new MersenneTwisterRNG();
    mock = new MockDataSet(rng, 50);
  }

  /**
   * Test the Weight part of the condition.
   * 
   */
  public void testWCondition() {
    int n = 100; // repeat the test n times

    // the dataline has all its attributes set to 0d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribut(EasyMock.anyInt())).andReturn(0d).atLeastOnce();
    EasyMock.replay(dl);

    // all the conditions are : attribut < 0
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
        if (rule.getW(condInd) < thr)
          assertTrue(rule.condition(condInd, dl));
        else
          assertFalse(rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

  /**
   * Test the Operator part of the condition, on numerical attributes
   * 
   */
  public void testOConditionNumerical() {
    int n = 100; // repeat the test n times

    // the dataline has all its attributes set to 1d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribut(EasyMock.anyInt())).andReturn(1d).atLeastOnce();
    EasyMock.replay(dl);

    for (int nloop = 0; nloop < n; nloop++) {
      mock.numericalDataset();

      CDRule rule = new CDRule(0.);
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        rule.setW(condInd, 1.); // all weights are 1 (active)
        rule.setO(condInd, rng.nextBoolean());
        rule.setV(condInd, 0);
      }

      // the condition is true if the operator is >=
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        if (rule.getO(condInd))
          assertTrue(rule.condition(condInd, dl));
        else
          assertFalse(rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

  /**
   * Test the Operator part of the condition, on numerical attributes
   * 
   */
  public void testOConditionCategorical() {
    int n = 100; // repeat the test n times

    // the dataline has all its attributes set to 1d
    DataLine dl = EasyMock.createMock(DataLine.class);
    EasyMock.expect(dl.getAttribut(EasyMock.anyInt())).andReturn(1d).atLeastOnce();
    EasyMock.replay(dl);

    Random rng = new MersenneTwisterRNG();
    for (int nloop = 0; nloop < n; nloop++) {
      mock.categoricalDataset();

      // all weights are 1 (active)
      CDRule rule = new CDRule(0.);
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        rule.setW(condInd, 1.);
        rule.setO(condInd, rng.nextBoolean());
        rule.setV(condInd, rng.nextInt(2)); // two categories
      }

      // the condition is true if the operator is == and the values are equal
      // (value==1), or the operator is != and the values are no equal
      // (value==0)
      for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
        if ((rule.getO(condInd) && rule.getV(condInd) == 1)
            || (!rule.getO(condInd) && rule.getV(condInd) != 1))
          assertTrue(rule.condition(condInd, dl));
        else
          assertFalse(rule.condition(condInd, dl));
      }

      mock.verify();
    }

    EasyMock.verify(dl);
  }

}
