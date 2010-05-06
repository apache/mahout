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

package org.apache.mahout.cf.taste.hadoop.similarity;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.mahout.cf.taste.impl.TasteTestCase;

public final class DistributedSimilarityTest extends TasteTestCase {

  public void testUncenteredZeroAssumingCosine() throws Exception {

    DistributedSimilarity similarity = new DistributedUncenteredZeroAssumingCosineSimilarity();

    assertSimilar(similarity, new Float[] { Float.NaN, Float.NaN, Float.NaN, Float.NaN, 1.0f },
        new Float[] { Float.NaN, 1.0f, 1.0f, 1.0f, 1.0f }, 0.5);

    assertSimilar(similarity, new Float[] { Float.NaN, 1.0f }, new Float[] { 1.0f, Float.NaN }, Double.NaN);
    assertSimilar(similarity, new Float[] { 1.0f, Float.NaN }, new Float[] { 1.0f, Float.NaN }, 1.0);
  }

  public void testPearsonCorrelation() throws Exception {

    DistributedSimilarity similarity = new DistributedPearsonCorrelationSimilarity();

    assertSimilar(similarity, new Float[] { 3.0f, -2.0f }, new Float[] { 3.0f, -2.0f }, 1.0);
    assertSimilar(similarity, new Float[] { 3.0f, 3.0f }, new Float[] { 3.0f, 3.0f }, Double.NaN);
    assertSimilar(similarity, new Float[] { Float.NaN, 3.0f }, new Float[] { 3.0f, Float.NaN }, Double.NaN);
  }

  private static void assertSimilar(DistributedSimilarity similarity,
                                    Float[] prefsX,
                                    Float[] prefsY,
                                    double expectedSimilarity) {

    double weightX = similarity.weightOfItemVector(Arrays.asList(prefsX).iterator());
    double weightY = similarity.weightOfItemVector(Arrays.asList(prefsY).iterator());

    List<CoRating> coRatings = new LinkedList<CoRating>();

    for (int n = 0; n < prefsX.length; n++) {
      Float x = prefsX[n];
      Float y = prefsY[n];

      if (!x.isNaN() && !y.isNaN()) {
        coRatings.add(new CoRating(x, y));
      }
    }

    double result = similarity.similarity(coRatings.iterator(), weightX, weightY);
    assertEquals(expectedSimilarity, result, EPSILON);
  }

}
