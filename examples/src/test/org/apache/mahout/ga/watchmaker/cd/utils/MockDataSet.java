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

import static org.easymock.classextension.EasyMock.expect;
import static org.easymock.classextension.EasyMock.createMock;
import static org.easymock.classextension.EasyMock.replay;
import static org.easymock.classextension.EasyMock.reset;

import java.util.Random;

import org.apache.mahout.ga.watchmaker.cd.DataSet;
import org.easymock.classextension.EasyMock;

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

    dataset = createMock(DataSet.class);
    DataSet.initialize(dataset);
  }

  /**
   * Generate a new dataset.
   * 
   * @param numRate numerical attributes rate.<br>
   *        0f : all attributes are categorical<br>
   *        1f : all attributes are numerical<br>
   *        otherwise : both numerical an categorical attributes are probable
   */
  public void randomDataset(float numRate) {
    reset(dataset);

    int nba = rng.nextInt(maxnba) + 1;
    expect(dataset.getNbAttributes()).andReturn(nba).anyTimes();

    // label at random position
    int labelpos = rng.nextInt(nba);
    expect(dataset.getLabelIndex()).andReturn(labelpos).anyTimes();

    for (int index = 0; index < nba; index++) {
      if (index == labelpos) {
        // two-classes
        prepareCategoricalAttribute(index, 2);
      } else if (rng.nextDouble() < numRate)
        prepareNumericalAttribute(index);
      else
        prepareCategoricalAttribute(index, rng.nextInt(100) + 1);
    }

    replay(dataset);
  }

  /**
   * Generate a new dataset. The attributes can be both numerical or
   * categorical.
   */
  public void randomDataset() {
    randomDataset(0.5f);
  }

  /**
   * Generate a new dataset. All the attributes are numerical.
   */
  public void numericalDataset() {
    randomDataset(1f);
  }

  /**
   * Generate a new dataset. All the attributes are categorical.
   */
  public void categoricalDataset() {
    randomDataset(0f);
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
    double max = rng.nextDouble() * (Float.MAX_VALUE - Float.MIN_VALUE)
        + Float.MIN_VALUE;
    double min = rng.nextDouble() * (max - Float.MIN_VALUE) + Float.MIN_VALUE;

    expect(dataset.isNumerical(index)).andReturn(true).anyTimes();
    expect(dataset.getMax(index)).andReturn(max).anyTimes();
    expect(dataset.getMin(index)).andReturn(min).anyTimes();
  }

  private void prepareCategoricalAttribute(int index, int nbcats) {
    expect(dataset.isNumerical(index)).andReturn(false).anyTimes();
    expect(dataset.getNbValues(index)).andReturn(nbcats).anyTimes();
  }

}
