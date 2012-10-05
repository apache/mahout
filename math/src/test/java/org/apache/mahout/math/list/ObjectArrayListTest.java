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

package org.apache.mahout.math.list;

import org.apache.mahout.math.MahoutTestCase;
import org.junit.Test;

/** tests for {@link ObjectArrayList}*/
public class ObjectArrayListTest extends MahoutTestCase {

  @Test
  public void emptyOnCreation() {
    ObjectArrayList<String> list = new ObjectArrayList<String>();
    assertTrue(list.isEmpty());
    assertEquals(0, list.size());
    list.add("1");
    list.add("2");
    list.add("3");
    assertEquals(3, list.size());
  }

  @Test
  public void correctSizeAfterInstantiation() {
    ObjectArrayList<String> list = new ObjectArrayList<String>(100);
    assertTrue(list.isEmpty());
    assertEquals(0, list.size());
  }

  @Test
  public void correctSizeAfterInstantiationWithElements() {
    ObjectArrayList<String> list = new ObjectArrayList<String>(new String[] { "1", "2", "3" });
    assertFalse(list.isEmpty());
    assertEquals(3, list.size());
  }

}
