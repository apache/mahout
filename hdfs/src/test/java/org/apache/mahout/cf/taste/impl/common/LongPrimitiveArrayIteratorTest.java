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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

import java.util.NoSuchElementException;

public final class LongPrimitiveArrayIteratorTest extends TasteTestCase {

  @Test(expected = NoSuchElementException.class)
  public void testEmpty() {
    LongPrimitiveIterator it = new LongPrimitiveArrayIterator(new long[0]);
    assertFalse(it.hasNext());
    it.next();
  }

  @Test(expected = NoSuchElementException.class)
  public void testNext() {
    LongPrimitiveIterator it = new LongPrimitiveArrayIterator(new long[] {3,2,1});
    assertTrue(it.hasNext());
    assertEquals(3, (long) it.next());
    assertTrue(it.hasNext());
    assertEquals(2, it.nextLong());
    assertTrue(it.hasNext());
    assertEquals(1, (long) it.next());    
    assertFalse(it.hasNext());
    it.nextLong();
  }

  @Test
  public void testPeekSkip() {
    LongPrimitiveIterator it = new LongPrimitiveArrayIterator(new long[] {3,2,1});
    assertEquals(3, it.peek());
    it.skip(2);
    assertEquals(1, it.nextLong());
    assertFalse(it.hasNext());
  }

}