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

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Random;

public final class CacheTest extends TasteTestCase {

  @Test
  public void testLotsOfGets() throws TasteException {
    Retriever<Object,Object> retriever = new IdentityRetriever();
    Cache<Object,Object> cache = new Cache<Object,Object>(retriever, 1000);
    for (int i = 0; i < 1000000; i++) {
      assertEquals(i, cache.get(i));
    }
  }

  @Test
  public void testMixedUsage() throws TasteException {
    Random random = RandomUtils.getRandom();
    Retriever<Object,Object> retriever = new IdentityRetriever();
    Cache<Object,Object> cache = new Cache<Object,Object>(retriever, 1000);
    for (int i = 0; i < 1000000; i++) {
      double r = random.nextDouble();
      if (r < 0.01) {
        cache.clear();
      } else if (r < 0.1) {
        cache.remove(r - 100);
      } else {
        assertEquals(i, cache.get(i));
      }
    }
  }
  
  private static class IdentityRetriever implements Retriever<Object,Object> {
    @Override
    public Object get(Object key) throws TasteException {
      return key;
    }
  }
}