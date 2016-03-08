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
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.util.Arrays;

import org.junit.Test;

public final class IntPairWritableTest extends MahoutTestCase {

  @Test
  public void testGetSet() {
    IntPairWritable n = new IntPairWritable();
    
    assertEquals(0, n.getFirst());
    assertEquals(0, n.getSecond());
    
    n.setFirst(5);
    n.setSecond(10);
    
    assertEquals(5, n.getFirst());
    assertEquals(10, n.getSecond());
    
    n = new IntPairWritable(2,4);
    
    assertEquals(2, n.getFirst());
    assertEquals(4, n.getSecond());
  }
  
  @Test
  public void testWritable() throws Exception {
    IntPairWritable one = new IntPairWritable(1,2);
    IntPairWritable two = new IntPairWritable(3,4);
    
    assertEquals(1, one.getFirst());
    assertEquals(2, one.getSecond());
    
    assertEquals(3, two.getFirst());
    assertEquals(4, two.getSecond());
    
    
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    DataOutput out = new DataOutputStream(bout);
    
    two.write(out);
    
    byte[] b = bout.toByteArray();
    
    ByteArrayInputStream bin = new ByteArrayInputStream(b);
    DataInput din = new DataInputStream(bin);
    
    one.readFields(din);
    
    assertEquals(two.getFirst(), one.getFirst());
    assertEquals(two.getSecond(), one.getSecond());    
  }
  
  @Test
  public void testComparable() {
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
      assertSame(input[expected[i]], sorted[i]);
    }
 
  }
}
