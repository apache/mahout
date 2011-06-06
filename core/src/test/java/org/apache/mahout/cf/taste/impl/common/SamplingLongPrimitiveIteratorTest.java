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

public final class SamplingLongPrimitiveIteratorTest extends TasteTestCase {

  @Test(expected = NoSuchElementException.class)
  public void testEmpty() {
    LongPrimitiveArrayIterator it = new LongPrimitiveArrayIterator(new long[0]);
    LongPrimitiveIterator sample = new SamplingLongPrimitiveIterator(it, 0.5);
    assertFalse(sample.hasNext());
    sample.next();
  }

  @Test(expected = NoSuchElementException.class)
  public void testNext() {
    LongPrimitiveArrayIterator it = new LongPrimitiveArrayIterator(new long[] {5,4,3,2,1});
    LongPrimitiveIterator sample = new SamplingLongPrimitiveIterator(it, 0.5);
    assertTrue(sample.hasNext());
    assertEquals(4, (long) sample.next());
    assertTrue(sample.hasNext());
    assertEquals(2, sample.nextLong());
    assertTrue(sample.hasNext());
    assertEquals(1, (long) sample.next());
    assertFalse(sample.hasNext());
    it.nextLong();
  }

  @Test
  public void testPeekSkip() {
    LongPrimitiveArrayIterator it = new LongPrimitiveArrayIterator(new long[] {8,7,6,5,4,3,2,1});
    LongPrimitiveIterator sample = new SamplingLongPrimitiveIterator(it, 0.5);
    assertEquals(7, sample.peek());
    sample.skip(1);
    assertEquals(4, sample.peek());
    assertTrue(sample.hasNext());
  }

}