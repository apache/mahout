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

package org.apache.mahout.cf.taste.hadoop.item;

import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;

public class IDReaderTest extends TasteTestCase {

  static final String USER_ITEM_FILTER_FIELD = "userItemFilter";

  @Test
  public void testUserItemFilter() throws Exception {
    Configuration conf = getConfiguration();
    IDReader idReader = new IDReader(conf);
    Map<Long, FastIDSet> userItemFilter = Maps.newHashMap();

    long user1 = 1;
    long user2 = 2;

    idReader.addUserAndItemIdToUserItemFilter(userItemFilter, user1, 100L);
    idReader.addUserAndItemIdToUserItemFilter(userItemFilter, user1, 200L);
    idReader.addUserAndItemIdToUserItemFilter(userItemFilter, user2, 300L);

    FastIDSet userIds = IDReader.extractAllUserIdsFromUserItemFilter(userItemFilter);

    assertEquals(2, userIds.size());
    assertTrue(userIds.contains(user1));
    assertTrue(userIds.contains(user1));

    setField(idReader, USER_ITEM_FILTER_FIELD, userItemFilter);

    FastIDSet itemsForUser1 = idReader.getItemsToRecommendForUser(user1);
    assertEquals(2, itemsForUser1.size());
    assertTrue(itemsForUser1.contains(100L));
    assertTrue(itemsForUser1.contains(200L));

    FastIDSet itemsForUser2 = idReader.getItemsToRecommendForUser(user2);
    assertEquals(1, itemsForUser2.size());
    assertTrue(itemsForUser2.contains(300L));

    FastIDSet itemsForNonExistingUser = idReader.getItemsToRecommendForUser(3L);
    assertTrue(itemsForNonExistingUser.isEmpty());
  }

}
