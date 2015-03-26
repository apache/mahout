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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.junit.Test;

import java.io.File;

public class FilePersistenceStrategyTest extends TasteTestCase {

  @Test
  public void persistAndLoad() throws Exception {
    FastByIDMap<Integer> userIDMapping = new FastByIDMap<Integer>();
    FastByIDMap<Integer> itemIDMapping = new FastByIDMap<Integer>();

    userIDMapping.put(123, 0);
    userIDMapping.put(456, 1);

    itemIDMapping.put(12, 0);
    itemIDMapping.put(34, 1);

    double[][] userFeatures = { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } };
    double[][] itemFeatures = { { 0.7, 0.8, 0.9 }, { 1.0, 1.1, 1.2 } };

    Factorization original = new Factorization(userIDMapping, itemIDMapping, userFeatures, itemFeatures);
    File storage = getTestTempFile("storage.bin");
    PersistenceStrategy persistenceStrategy = new FilePersistenceStrategy(storage);

    assertNull(persistenceStrategy.load());

    persistenceStrategy.maybePersist(original);
    Factorization clone = persistenceStrategy.load();

    assertEquals(original, clone);
  }
}
