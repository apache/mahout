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

/** <p>Tests {@link ThresholdUserNeighborhood}.</p> */
public final class ThresholdNeighborhoodTest extends TasteTestCase {

  public void testNeighborhood() throws Exception {

    DataModel dataModel = getDataModel();

    long[] neighborhood =
        new ThresholdUserNeighborhood(1.0, new DummySimilarity(dataModel), dataModel).getUserNeighborhood(1);
    assertNotNull(neighborhood);
    assertTrue(neighborhood.length == 0);

    long[] neighborhood2 =
        new ThresholdUserNeighborhood(0.8, new DummySimilarity(dataModel), dataModel).getUserNeighborhood(1);
    assertNotNull(neighborhood2);
    assertEquals(1, neighborhood2.length);
    assertTrue(arrayContains(neighborhood2, 2));

    long[] neighborhood3 =
        new ThresholdUserNeighborhood(0.6, new DummySimilarity(dataModel), dataModel).getUserNeighborhood(2);
    assertNotNull(neighborhood3);
    assertEquals(3, neighborhood3.length);
    assertTrue(arrayContains(neighborhood3, 1));
    assertTrue(arrayContains(neighborhood3, 3));
    assertTrue(arrayContains(neighborhood3, 4));


  }

}
