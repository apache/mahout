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

import java.util.LinkedList;
import java.util.List;

import org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob;
import org.apache.mahout.cf.taste.impl.TasteTestCase;

/**
 * base testcase for all tests for implementations of {@link DistributedItemSimilarity}
 */
public abstract class DistributedItemSimilarityTestCase extends TasteTestCase {

  /**
   * emulates the way the similarity would be computed by {@link ItemSimilarityJob}
   *
   * @param similarity
   * @param numberOfUsers
   * @param prefsX
   * @param prefsY
   * @param expectedSimilarity
   */
  protected static void assertSimilar(DistributedItemSimilarity similarity,
                                    int numberOfUsers,
                                    Float[] prefsX,
                                    Float[] prefsY,
                                    double expectedSimilarity) {

    List<Float> nonNaNPrefsX = new LinkedList<Float>();
    for (Float prefX : prefsX) {
      if (!prefX.isNaN()) {
        nonNaNPrefsX.add(prefX);
      }
    }

    List<Float> nonNaNPrefsY = new LinkedList<Float>();
    for (Float prefY : prefsY) {
      if (!prefY.isNaN()) {
        nonNaNPrefsY.add(prefY);
      }
    }

    double weightX = similarity.weightOfItemVector(nonNaNPrefsX.iterator());
    double weightY = similarity.weightOfItemVector(nonNaNPrefsY.iterator());

    List<CoRating> coRatings = new LinkedList<CoRating>();

    for (int n = 0; n < prefsX.length; n++) {
      Float x = prefsX[n];
      Float y = prefsY[n];

      if (!x.isNaN() && !y.isNaN()) {
        coRatings.add(new CoRating(x, y));
      }
    }

    double result = similarity.similarity(coRatings.iterator(), weightX, weightY, numberOfUsers);
    assertEquals(expectedSimilarity, result, EPSILON);
  }

}
