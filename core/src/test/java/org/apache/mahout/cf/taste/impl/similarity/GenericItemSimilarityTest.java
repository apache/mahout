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

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.util.ArrayList;
import java.util.List;

/** <p>Tests {@link GenericItemSimilarity}.</p> */
public final class GenericItemSimilarityTest extends SimilarityTestCase {

  public void testSimple() {
    Comparable<?> item1 = "1";
    Comparable<?> item2 = "2";
    Comparable<?> item3 = "3";
    Comparable<?> item4 = "4";
    List<GenericItemSimilarity.ItemItemSimilarity> similarities =
        new ArrayList<GenericItemSimilarity.ItemItemSimilarity>(4);
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(item1, item2, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(item2, item1, 0.6));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(item1, item1, 0.5));
    similarities.add(new GenericItemSimilarity.ItemItemSimilarity(item1, item3, 0.3));
    GenericItemSimilarity itemCorrelation = new GenericItemSimilarity(similarities);
    assertEquals(1.0, itemCorrelation.itemSimilarity(item1, item1));
    assertEquals(0.6, itemCorrelation.itemSimilarity(item1, item2));
    assertEquals(0.6, itemCorrelation.itemSimilarity(item2, item1));
    assertEquals(0.3, itemCorrelation.itemSimilarity(item1, item3));
    assertTrue(Double.isNaN(itemCorrelation.itemSimilarity(item3, item4)));
  }

  public void testFromCorrelation() throws Exception {
    DataModel dataModel = getDataModel(
            new Comparable<?>[] {"test1", "test2", "test3"},
            new Double[][] {
                    {1.0, 2.0},
                    {2.0, 5.0},
                    {3.0, 6.0},
            });
    ItemSimilarity otherSimilarity = new PearsonCorrelationSimilarity(dataModel);
    ItemSimilarity itemSimilarity = new GenericItemSimilarity(otherSimilarity, dataModel);
    assertCorrelationEquals(1.0, itemSimilarity.itemSimilarity("0", "0"));
    assertCorrelationEquals(0.960768922830523, itemSimilarity.itemSimilarity("0", "1"));
  }

}
