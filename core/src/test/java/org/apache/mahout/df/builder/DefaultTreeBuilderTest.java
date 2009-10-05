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

package org.apache.mahout.df.builder;

import java.util.Random;

import org.apache.commons.lang.ArrayUtils;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.Utils;

import junit.framework.TestCase;

public class DefaultTreeBuilderTest extends TestCase {

  public void testRandomAttributes() throws Exception {
    Random rng = RandomUtils.getRandom();
    int maxNbAttributes = 100;
    int n = 100;

    for (int nloop = 0; nloop < n; nloop++) {
      int nbAttributes = rng.nextInt(maxNbAttributes) + 1;

      // generate a small data, only to get the dataset
      Data data = Utils.randomData(rng, nbAttributes, 1);
      if (data.getDataset().nbAttributes() == 0)
        continue;

      int m = rng.nextInt(data.getDataset().nbAttributes()) + 1;

      int[] attrs = DefaultTreeBuilder.randomAttributes(data.getDataset(), rng, m);

      assertEquals(m, attrs.length);

      for (int index = 0; index < m; index++) {
        int attr = attrs[index];

        // each attribute should be in the range [0, nbAttributes[
        assertTrue(attr >= 0);
        assertTrue(attr < nbAttributes);

        // each attribute should appear only once
        assertEquals(index, ArrayUtils.lastIndexOf(attrs, attr));
      }
    }
  }
}
