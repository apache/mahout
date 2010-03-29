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

package org.apache.mahout.utils.nlp.collocations.llr;

import junit.framework.Assert;

import org.junit.Test;

public class GramKeyGroupComparatorTest {

  @Test
  public void testComparator() {
    byte[] foo   = new byte[1];
    foo[0] = (byte) 1;

    byte[] empty = new byte[0];
    GramKey a = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), empty); // base
    GramKey b = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), foo);   // vary byte
    GramKey c = new GramKey(new Gram("foo", 2, Gram.Type.HEAD), empty); // vary freq
    GramKey d = new GramKey(new Gram("foo", 1, Gram.Type.TAIL), empty); // vary type
    GramKey e = new GramKey(new Gram("bar", 5, Gram.Type.HEAD), empty); // vary string
    
    GramKeyGroupComparator cmp = new GramKeyGroupComparator();

    Assert.assertEquals(0, cmp.compare(a, b));
    Assert.assertEquals(0, cmp.compare(a, c));
    Assert.assertTrue(0 > cmp.compare(a, d));
    Assert.assertTrue(0 < cmp.compare(a, e));
    Assert.assertTrue(0 < cmp.compare(d, e));
  }
}
