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

package org.apache.mahout.ga.watchmaker.cd.utils;

import org.apache.mahout.ga.watchmaker.cd.DataSet;
import org.easymock.classextension.EasyMock;

import java.util.Random;

/**
 * Generate a mock dataset using EasyMock. The dataset contains a random number
 * of attributes. Each attribute can be numerical or categorical (choosen
 * randomly).
 */
public class MockDataSet {

  private Random rng;

  private int maxnba;

  private DataSet dataset;

  /**
   * 
   * @param maxnba max number of attributes
   */
  public MockDataSet(Random rng, int maxnba) {
    assert maxnba > 0 : "maxnba must be greater than 0";

    this.rng = rng;
    this.maxnba = maxnba;

    dataset = EasyMock.createMock(DataSet.class);
    DataSet.initialize(dataset);
  }

  /**
   * Generate a new dataset.
   * 
   * @param numRate numerical attributes rate.<br>
   *        0.0 : all attributes are categorical<br>
   *        1.0 : all attributes are numerical<br>
   *        otherwise : both numerical an categorical attributes are probable
   */
  public void randomDataset(double numRate) {
    EasyMock.reset(dataset);

    int nba = rng.nextInt(maxnba) + 1;
    EasyMock.expect(dataset.getNbAttributes()).andReturn(nba).anyTimes();

    // label at random position
    int labelpos = rng.nextInt(nba);
    EasyMock.expect(dataset.getLabelIndex()).andReturn(labelpos).anyTimes();

    for (int index = 0; index < nba; index++) {
      if (index == labelpos) {
        // two-classes
        prepareCategoricalAttribute(index, 2);
      } else if (rng.nextDouble() < numRate)
        prepareNumericalAttribute(index);
      else
        prepareCategoricalAttribute(index, rng.nextInt(100) + 1);
    }

    EasyMock.replay(dataset);
  }

  /**
   * Generate a new dataset. The attributes can be both numerical or
   * categorical.
   */
  public void randomDataset() {
    randomDataset(0.5);
  }

  /**
   * Generate a new dataset. All the attributes are numerical.
   */
  public void numericalDataset() {
    randomDataset(1.0);
  }

  /**
   * Generate a new dataset. All the attributes are categorical.
   */
  public void categoricalDataset() {
    randomDataset(0.0);
  }

  /**
   * Verifies the dataset mock object.
   * 
   * @see org.easymock.classextension.EasyMock#verify(Object...)
   */
  public void verify() {
    EasyMock.verify(dataset);
  }

  private void prepareNumericalAttribute(int index) {

    // srowen: I 'fixed' this to not use Double.{MAX,MIN}_VALUE since
    // it does not seem like that has the desired effect 
    double max = rng.nextDouble() * ((long) Integer.MAX_VALUE - Integer.MIN_VALUE) + Integer.MIN_VALUE;
    double min = rng.nextDouble() * (max - Integer.MIN_VALUE) + Integer.MIN_VALUE;

    EasyMock.expect(dataset.isNumerical(index)).andReturn(true).anyTimes();
    EasyMock.expect(dataset.getMax(index)).andReturn(max).anyTimes();
    EasyMock.expect(dataset.getMin(index)).andReturn(min).anyTimes();
  }

  private void prepareCategoricalAttribute(int index, int nbcats) {
    EasyMock.expect(dataset.isNumerical(index)).andReturn(false).anyTimes();
    EasyMock.expect(dataset.getNbValues(index)).andReturn(nbcats).anyTimes();
  }

}
