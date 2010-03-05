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

package org.apache.mahout.common;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

import junit.framework.Assert;

import org.junit.Test;

public class IntPairWritableTest {

  @Test
  public void testGetSet() {
    IntPairWritable n = new IntPairWritable();
    
    Assert.assertEquals(0, n.getFirst());
    Assert.assertEquals(0, n.getSecond());
    
    n.setFirst(5);
    n.setSecond(10);
    
    Assert.assertEquals(5, n.getFirst());
    Assert.assertEquals(10, n.getSecond());
    
    n = new IntPairWritable(2,4);
    
    Assert.assertEquals(2, n.getFirst());
    Assert.assertEquals(4, n.getSecond());
  }
  
  @Test
  public void testWritable() throws IOException {
    IntPairWritable one = new IntPairWritable(1,2);
    IntPairWritable two = new IntPairWritable(3,4);
    
    Assert.assertEquals(1, one.getFirst());
    Assert.assertEquals(2, one.getSecond());
    
    Assert.assertEquals(3, two.getFirst());
    Assert.assertEquals(4, two.getSecond());
    
    
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(bout);
    
    two.write(out);
    
    byte[] b = bout.toByteArray();
    
    ByteArrayInputStream bin = new ByteArrayInputStream(b);
    DataInputStream din = new DataInputStream(bin);
    
    one.readFields(din);
    
    Assert.assertEquals(two.getFirst(), one.getFirst());
    Assert.assertEquals(two.getSecond(), one.getSecond());    
  }
  
  @Test
  public void testComparable() throws IOException {
    IntPairWritable[] input = {
        new IntPairWritable(2,3),
        new IntPairWritable(2,2),
        new IntPairWritable(1,3),
        new IntPairWritable(1,2),
        new IntPairWritable(2,1),
        new IntPairWritable(2,2),
        new IntPairWritable(1,-2),
        new IntPairWritable(1,-1),
        new IntPairWritable(-2,-2),
        new IntPairWritable(-2,-1),
        new IntPairWritable(-1,-1),
        new IntPairWritable(-1,-2),
        new IntPairWritable(Integer.MAX_VALUE,1),
        new IntPairWritable(Integer.MAX_VALUE/2,1),
        new IntPairWritable(Integer.MIN_VALUE,1),
        new IntPairWritable(Integer.MIN_VALUE/2,1)
        
    };
    
    IntPairWritable[] sorted = new IntPairWritable[input.length];
    System.arraycopy(input, 0, sorted, 0, input.length);
    Arrays.sort(sorted);
    
    int[] expected = {
        14, 15, 8, 9, 11, 10, 6, 7, 3, 2, 4, 1, 5, 0, 13, 12
    };
    
    for (int i=0; i < input.length; i++) {
      Assert.assertSame(input[expected[i]], sorted[i]);
    }
 
  }
}
