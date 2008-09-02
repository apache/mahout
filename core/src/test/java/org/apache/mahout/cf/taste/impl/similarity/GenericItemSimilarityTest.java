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

import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>Tests {@link GenericItemSimilarity}.</p>
 */
public final class GenericItemSimilarityTest extends SimilarityTestCase {

  public void testSimple() {
    Item item1 = new GenericItem<String>("1");
    Item item2 = new GenericItem<String>("2");
    Item item3 = new GenericItem<String>("3");
    Item item4 = new GenericItem<String>("4");
    List<GenericItemSimilarity.ItemItemCorrelation> correlations =
            new ArrayList<GenericItemSimilarity.ItemItemCorrelation>(4);
    correlations.add(new GenericItemSimilarity.ItemItemCorrelation(item1, item2, 0.5));
    correlations.add(new GenericItemSimilarity.ItemItemCorrelation(item2, item1, 0.6));
    correlations.add(new GenericItemSimilarity.ItemItemCorrelation(item1, item1, 0.5));
    correlations.add(new GenericItemSimilarity.ItemItemCorrelation(item1, item3, 0.3));
    GenericItemSimilarity itemCorrelation = new GenericItemSimilarity(correlations);
    assertEquals(1.0, itemCorrelation.itemCorrelation(item1, item1));
    assertEquals(0.6, itemCorrelation.itemCorrelation(item1, item2));
    assertEquals(0.6, itemCorrelation.itemCorrelation(item2, item1));
    assertEquals(0.3, itemCorrelation.itemCorrelation(item1, item3));
    assertTrue(Double.isNaN(itemCorrelation.itemCorrelation(item3, item4)));
  }

  public void testFromCorrelation() throws Exception {
    User user1 = getUser("test1", 1.0, 2.0);
    User user2 = getUser("test2", 2.0, 5.0);
    User user3 = getUser("test3", 3.0, 6.0);
    DataModel dataModel = getDataModel(user1, user2, user3);
    ItemSimilarity otherSimilarity = new PearsonCorrelationSimilarity(dataModel);
    ItemSimilarity itemSimilarity = new GenericItemSimilarity(otherSimilarity, dataModel);
    assertCorrelationEquals(1.0,
                            itemSimilarity.itemCorrelation(dataModel.getItem("0"), dataModel.getItem("0")));
    assertCorrelationEquals(0.960768922830523,
                            itemSimilarity.itemCorrelation(dataModel.getItem("0"), dataModel.getItem("1")));
  }

}
