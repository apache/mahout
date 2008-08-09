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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * <p>Tests {@link NearestNUserNeighborhood}.</p>
 */
public final class NearestNNeighborhoodTest extends TasteTestCase {

  public void testNeighborhood() throws Exception {

    List<User> users = getMockUsers();
    DataModel dataModel = new GenericDataModel(users);

    Collection<User> neighborhood =
            new NearestNUserNeighborhood(1, new DummyCorrelation(), dataModel).getUserNeighborhood("test1");
    assertNotNull(neighborhood);
    assertEquals(1, neighborhood.size());
    assertTrue(neighborhood.contains(users.get(1)));

    Collection<User> neighborhood2 =
            new NearestNUserNeighborhood(2, new DummyCorrelation(), dataModel).getUserNeighborhood("test2");
    assertNotNull(neighborhood2);
    assertEquals(2, neighborhood2.size());
    assertTrue(neighborhood2.contains(users.get(0)));
    assertTrue(neighborhood2.contains(users.get(2)));

    Collection<User> neighborhood3 =
            new NearestNUserNeighborhood(4, new DummyCorrelation(), dataModel).getUserNeighborhood("test4");
    assertNotNull(neighborhood3);
    assertEquals(3, neighborhood3.size());
    assertTrue(neighborhood3.contains(users.get(0)));
    assertTrue(neighborhood3.contains(users.get(1)));
    assertTrue(neighborhood3.contains(users.get(2)));

  }

  public void testRefresh() throws Exception {
    // Make sure this doesn't throw an exception
    DataModel dataModel = new GenericDataModel(Collections.singletonList(getUser("test1", 0.1)));
    new NearestNUserNeighborhood(1, new DummyCorrelation(), dataModel).refresh(null);
  }

}
