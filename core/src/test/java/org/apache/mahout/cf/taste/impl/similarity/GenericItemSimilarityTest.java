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
package org.apache.mahout.cf.taste.impl.similarity;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/** <p>Tests {@link GenericItemSimilarity}.</p> */
public final class GenericItemSimilarityTest extends SimilarityTestCase {

  @Test
  public void testSimple() {
    List<GenericItemSimilarity.ItemItemSimilarity> similarities = Lists.newArrayList();
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 2, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(2, 1, 0.6));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 1, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(1, 3, 0.3));
    GenericItemSimilarity itemCorrelation = new GenericItemSimilarity(similarities);
    assertEquals(1.0, itemCorrelation.itemSimilarity(1, 1), EPSILON);
    assertEquals(0.6, itemCorrelation.itemSimilarity(1, 2), EPSILON);
    assertEquals(0.6, itemCorrelation.itemSimilarity(2, 1), EPSILON);
    assertEquals(0.3, itemCorrelation.itemSimilarity(1, 3), EPSILON);
    assertTrue(Double.isNaN(itemCorrelation.itemSimilarity(3, 4)));
  }

  @Test
  public void testFromCorrelation() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {1.0, 2.0},
                    {2.0, 5.0},
                    {3.0, 6.0},
            });
    ItemSimilarity otherSimilarity = new PearsonCorrelationSimilarity(dataModel);
    ItemSimilarity itemSimilarity = new GenericItemSimilarity(otherSimilarity, dataModel);
    assertCorrelationEquals(1.0, itemSimilarity.itemSimilarity(0, 0));
    assertCorrelationEquals(0.960768922830523, itemSimilarity.itemSimilarity(0, 1));
  }

  @Test
  public void testAllSimilaritiesWithoutIndex() throws TasteException {

    List<GenericItemSimilarity.ItemItemSimilarity> itemItemSimilarities =
        Arrays.asList(new GenericItemSimilarity.ItemItemSimilarity(1L, 2L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(1L, 3L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(2L, 1L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(3L, 5L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(3L, 4L, 0.2));

    ItemSimilarity similarity = new GenericItemSimilarity(itemItemSimilarities);

    assertTrue(containsExactly(similarity.allSimilarItemIDs(1L), 2L, 3L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(2L), 1L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(3L), 1L, 5L, 4L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(4L), 3L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(5L), 3L));
  }

  @Test
  public void testAllSimilaritiesWithIndex() throws TasteException {

    List<GenericItemSimilarity.ItemItemSimilarity> itemItemSimilarities =
        Arrays.asList(new GenericItemSimilarity.ItemItemSimilarity(1L, 2L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(1L, 3L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(2L, 1L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(3L, 5L, 0.2),
                      new GenericItemSimilarity.ItemItemSimilarity(3L, 4L, 0.2));

    ItemSimilarity similarity = new GenericItemSimilarity(itemItemSimilarities);

    assertTrue(containsExactly(similarity.allSimilarItemIDs(1L), 2L, 3L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(2L), 1L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(3L), 1L, 5L, 4L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(4L), 3L));
    assertTrue(containsExactly(similarity.allSimilarItemIDs(5L), 3L));
  }

  private static boolean containsExactly(long[] allIDs, long... shouldContainID) {
    return new FastIDSet(allIDs).intersectionSize(new FastIDSet(shouldContainID)) == shouldContainID.length;
  }

}
