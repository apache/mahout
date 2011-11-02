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

package org.apache.mahout.classifier.df.split;

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Utils;
import org.junit.Test;

public final class OptIgSplitTest extends MahoutTestCase {

  private static final int NUM_ATTRIBUTES = 20;

  private static final int NUM_INSTANCES = 100;

  @Test
  public void testComputeSplit() throws Exception {
    IgSplit ref = new DefaultIgSplit();
    IgSplit opt = new OptIgSplit();

    Random rng = RandomUtils.getRandom();
    Data data = Utils.randomData(rng, NUM_ATTRIBUTES, false, NUM_INSTANCES);

    for (int nloop = 0; nloop < 100; nloop++) {
      int attr = rng.nextInt(data.getDataset().nbAttributes());
      // System.out.println("IsNumerical: " + data.dataset.isNumerical(attr));

      Split expected = ref.computeSplit(data, attr);
      Split actual = opt.computeSplit(data, attr);

      assertEquals(expected.getIg(), actual.getIg(), EPSILON);
      assertEquals(expected.getSplit(), actual.getSplit(), EPSILON);
    }
  }

}
