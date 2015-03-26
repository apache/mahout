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

package org.apache.mahout.classifier.df.builder;

import java.lang.reflect.Method;
import java.util.Random;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

public final class DecisionTreeBuilderTest extends MahoutTestCase {

  /**
   * make sure that DecisionTreeBuilder.randomAttributes() returns the correct number of attributes, that have not been
   * selected yet
   */
  @Test
  public void testRandomAttributes() throws Exception {
    Random rng = RandomUtils.getRandom();
    int nbAttributes = rng.nextInt(100) + 1;
    boolean[] selected = new boolean[nbAttributes];

    for (int nloop = 0; nloop < 100; nloop++) {
      Arrays.fill(selected, false);

      // randomly select some attributes
      int nbSelected = rng.nextInt(nbAttributes - 1);
      for (int index = 0; index < nbSelected; index++) {
        int attr;
        do {
          attr = rng.nextInt(nbAttributes);
        } while (selected[attr]);

        selected[attr] = true;
      }

      int m = rng.nextInt(nbAttributes);

      Method randomAttributes = DecisionTreeBuilder.class.getDeclaredMethod("randomAttributes",
        Random.class, boolean[].class, int.class);
      randomAttributes.setAccessible(true);
      int[] attrs = (int[]) randomAttributes.invoke(null, rng, selected, m);

      assertNotNull(attrs);
      assertEquals(Math.min(m, nbAttributes - nbSelected), attrs.length);

      for (int attr : attrs) {
        // the attribute should not be already selected
        assertFalse("an attribute has already been selected", selected[attr]);

        // each attribute should be in the range [0, nbAttributes[
        assertTrue(attr >= 0);
        assertTrue(attr < nbAttributes);

        // each attribute should appear only once
        assertEquals(ArrayUtils.indexOf(attrs, attr), ArrayUtils.lastIndexOf(attrs, attr));
      }
    }
  }
}
