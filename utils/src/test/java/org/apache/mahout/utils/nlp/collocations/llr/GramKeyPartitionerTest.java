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


public class GramKeyPartitionerTest {
  @Test
  public void testPartition() {
    byte[] foo = new byte[1];
    foo[0] = 1;

    foo[0] = 2;

    byte[] empty = new byte[0];
    GramKey a = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), empty);
    GramKey b = new GramKey(new Gram("foo", 1, Gram.Type.HEAD), foo);
    byte[] bar = new byte[1];
    GramKey c = new GramKey(new Gram("foo", 2, Gram.Type.HEAD), bar);
    GramKey d = new GramKey(new Gram("foo", 1, Gram.Type.TAIL), empty);
    GramKey e = new GramKey(new Gram("foo", 2, Gram.Type.TAIL), foo);
    
    GramKeyPartitioner p = new GramKeyPartitioner();
    int numPartitions = 5;
    
    int ap = p.getPartition(a, null, numPartitions);
    int bp = p.getPartition(b, null, numPartitions);
    int cp = p.getPartition(c, null, numPartitions);
    int dp = p.getPartition(d, null, numPartitions);
    int ep = p.getPartition(e, null, numPartitions);
    
    Assert.assertEquals(ap, bp);
    Assert.assertEquals(ap, cp);
    Assert.assertEquals(dp, ep);
  }
}
