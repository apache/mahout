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

package org.apache.mahout.vectorizer.collocations.llr;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class GramKeyGroupComparatorTest extends MahoutTestCase {

  @Test
  public void testComparator() {
    byte[] foo   = new byte[1];
    foo[0] = 1;

    byte[] empty = new byte[0];
    GramKey a = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), empty); // base
    GramKey b = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), foo);   // vary byte
    GramKey c = new GramKey(new Gram("foo", 2, Gram.Type.HEAD), empty); // vary freq
    GramKey d = new GramKey(new Gram("foo", 1, Gram.Type.TAIL), empty); // vary type
    GramKey e = new GramKey(new Gram("bar", 5, Gram.Type.HEAD), empty); // vary string
    
    GramKeyGroupComparator cmp = new GramKeyGroupComparator();

    assertEquals(0, cmp.compare(a, b));
    assertEquals(0, cmp.compare(a, c));
    assertTrue(cmp.compare(a, d) < 0);
    assertTrue(cmp.compare(a, e) > 0);
    assertTrue(cmp.compare(d, e) > 0);
  }
}
