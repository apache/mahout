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

/** <p>Tests {@link ThresholdUserNeighborhood}.</p> */
public final class ThresholdNeighborhoodTest extends TasteTestCase {

  public void testNeighborhood() throws Exception {

    DataModel dataModel = getDataModel();

    Collection<Comparable<?>> neighborhood =
        new ThresholdUserNeighborhood(19.0, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test1");
    assertNotNull(neighborhood);
    assertTrue(neighborhood.isEmpty());

    Collection<Comparable<?>> neighborhood2 =
        new ThresholdUserNeighborhood(9.0, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test1");
    assertNotNull(neighborhood2);
    assertEquals(1, neighborhood2.size());
    assertTrue(neighborhood2.contains("test2"));

    Collection<Comparable<?>> neighborhood3 =
        new ThresholdUserNeighborhood(0.9, new DummySimilarity(dataModel), dataModel).getUserNeighborhood("test2");
    assertNotNull(neighborhood3);
    assertEquals(3, neighborhood3.size());
    assertTrue(neighborhood3.contains("test1"));
    assertTrue(neighborhood3.contains("test3"));
    assertTrue(neighborhood3.contains("test4"));

  }

}
