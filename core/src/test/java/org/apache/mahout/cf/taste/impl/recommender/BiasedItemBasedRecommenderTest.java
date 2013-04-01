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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.math.Sorting;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BiasedItemBasedRecommenderTest {

  @Test
  public void sorting() {

    double[] similarities = new double[] { 0.1, 1.0, 0.5 };
    float[] ratings = new float[] { 3, 1, 2 };
    long[] itemIDs = new long[] { 3, 1, 2 };

    Sorting.quickSort(0, similarities.length, new BiasedItemBasedRecommender.SimilaritiesComparator(similarities),
        new BiasedItemBasedRecommender.SimilaritiesRatingsItemIDsSwapper(similarities, ratings, itemIDs));

    assertEquals(1d, similarities[0], 0d);
    assertEquals(0.5d, similarities[1], 0d);
    assertEquals(0.1d, similarities[2], 0d);

    assertEquals(1f, ratings[0], 0f);
    assertEquals(2f, ratings[1], 0f);
    assertEquals(3f, ratings[2], 0f);

    assertEquals(1l, itemIDs[0]);
    assertEquals(2l, itemIDs[1]);
    assertEquals(3l, itemIDs[2]);
  }

}
