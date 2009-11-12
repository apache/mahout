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

package org.apache.mahout.df.split;

import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.Utils;

import junit.framework.TestCase;

public class OptIgSplitTest extends TestCase {

  protected static final int nbAttributes = 20;

  protected static final int numInstances = 100;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
  }
 
  public void testComputeSplit() throws Exception {
    int n = 100;

    IgSplit ref = new DefaultIgSplit();
    IgSplit opt = new OptIgSplit();

    Random rng = RandomUtils.getRandom();
    Data data = Utils.randomData(rng, nbAttributes, numInstances);

    for (int nloop = 0; nloop < n; nloop++) {
      int attr = rng.nextInt(data.getDataset().nbAttributes());
      // System.out.println("IsNumerical: " + data.dataset.isNumerical(attr));

      Split expected = ref.computeSplit(data, attr);
      Split actual = opt.computeSplit(data, attr);

      assertEquals(expected.ig, actual.ig);
      assertEquals(expected.split, actual.split);
    }
  }

}
