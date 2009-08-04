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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;

import java.util.Collection;

/** <p>Tests {@link NearestNUserNeighborhood}.</p> */
public final class NearestNNeighborhoodTest extends TasteTestCase {

  public void testNeighborhood() throws Exception {

    DataModel dataModel = getDataModel();

    Collection<Comparable<?>> neighborhood =
        new NearestNUserNeighborhood(1, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test1");
    assertNotNull(neighborhood);
    assertEquals(1, neighborhood.size());
    assertTrue(neighborhood.contains("test2"));

    Collection<Comparable<?>> neighborhood2 =
        new NearestNUserNeighborhood(2, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test2");
    assertNotNull(neighborhood2);
    assertEquals(2, neighborhood2.size());
    assertTrue(neighborhood2.contains("test1"));
    assertTrue(neighborhood2.contains("test3"));

    Collection<Comparable<?>> neighborhood3 =
        new NearestNUserNeighborhood(4, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test4");
    assertNotNull(neighborhood3);
    assertEquals(3, neighborhood3.size());
    assertTrue(neighborhood3.contains("test1"));
    assertTrue(neighborhood3.contains("test2"));
    assertTrue(neighborhood3.contains("test3"));

  }

}
