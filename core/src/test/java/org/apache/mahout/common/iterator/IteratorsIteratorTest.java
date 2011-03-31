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

package org.apache.mahout.common.iterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class IteratorsIteratorTest extends MahoutTestCase {

  @Test
  public void testEmpty() {
    assertFalse(new IteratorsIterator<Object>(Collections.<Iterator<Object>>emptyList()).hasNext());
  }

  @Test
  public void testSequences() {
    Iterator<Integer> it = new IteratorsIterator<Integer>(Arrays.asList(
      Integers.iterator(3), Integers.iterator(0), Integers.iterator(1)
    ));
    assertTrue(it.hasNext());
    assertEquals(0, (int) it.next());
    assertEquals(1, (int) it.next());
    assertEquals(2, (int) it.next());
    assertEquals(0, (int) it.next());
    assertFalse(it.hasNext());
  }

}
