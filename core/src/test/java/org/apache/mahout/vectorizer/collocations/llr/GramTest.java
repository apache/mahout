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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.HashMap;

import com.google.common.collect.Maps;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class GramTest extends MahoutTestCase {
  
  @Test
  public void testConstructorsGetters() {
    Gram one = new Gram("foo", 2, Gram.Type.HEAD);
    
    assertEquals("foo", one.getString());
    assertEquals(2, one.getFrequency());
    assertEquals(Gram.Type.HEAD, one.getType());
    
    Gram oneClone = new Gram(one);
    
    assertEquals("foo", oneClone.getString());
    assertEquals(2, oneClone.getFrequency());
    assertEquals(Gram.Type.HEAD, oneClone.getType());
    
    Gram two = new Gram("foo", 3, Gram.Type.TAIL);
    assertEquals(Gram.Type.TAIL, two.getType());
    
    Gram three = new Gram("foo", 4, Gram.Type.UNIGRAM);
    assertEquals(Gram.Type.UNIGRAM, three.getType());
    
    Gram four = new Gram("foo", 5, Gram.Type.NGRAM);
    assertEquals(Gram.Type.NGRAM, four.getType());
  }

  @Test(expected = NullPointerException.class)
  public void testNull1() {
    new Gram(null, 4, Gram.Type.UNIGRAM);
  }

  @Test(expected = NullPointerException.class)
  public void testNull2() {
    new Gram("foo", 4, null);
  }
  
  @Test
  public void testEquality() {
    Gram one = new Gram("foo", 2, Gram.Type.HEAD);  
    Gram two = new Gram("foo", 3, Gram.Type.HEAD);

    assertEquals(one, two);
    assertEquals(two, one);
    
    Gram three = new Gram("foo", 4, Gram.Type.TAIL);
    Gram four = new Gram("foo", Gram.Type.UNIGRAM);
    
    assertTrue(!three.equals(two));
    assertTrue(!four.equals(one));
    assertTrue(!one.equals(four));
    
    Gram five = new Gram("foo", 5, Gram.Type.UNIGRAM);

    assertEquals(four, five);
    
    Gram six = new Gram("foo", 6, Gram.Type.NGRAM);
    Gram seven = new Gram("foo", 7, Gram.Type.NGRAM);
    
    assertTrue(!five.equals(six));
    assertEquals(six, seven);
    
    Gram eight = new Gram("foobar", 4, Gram.Type.TAIL);
    
    assertTrue(!eight.equals(four));
    assertTrue(!eight.equals(three));
    assertTrue(!eight.equals(two));
    assertTrue(!eight.equals(one));
  }
  
  @Test
  public void testHashing() {
    Gram[] input =
    {
     new Gram("foo", 2, Gram.Type.HEAD),
     new Gram("foo", 3, Gram.Type.HEAD),
     new Gram("foo", 4, Gram.Type.TAIL),
     new Gram("foo", 5, Gram.Type.TAIL),
     new Gram("bar", 6, Gram.Type.HEAD),
     new Gram("bar", 7, Gram.Type.TAIL),
     new Gram("bar", 8, Gram.Type.NGRAM),
     new Gram("bar", Gram.Type.UNIGRAM)
    };
    
    HashMap<Gram,Gram> map = Maps.newHashMap();
    for (Gram n : input) {
      Gram val = map.get(n);
      if (val != null) {
        val.incrementFrequency(n.getFrequency());
      } else {
        map.put(n, n);
      }
    }
    
    // frequencies of the items in the map.
    int[] freq = {
                  5,
                  3,
                  9,
                  5,
                  6,
                  7,
                  8,
                  1
    };
    
    // true if the index should be the item in the map
    boolean[] memb = {
                      true,
                      false,
                      true,
                      false,
                      true,
                      true,
                      true,
                      true
    };
    
    for (int i = 0; i < input.length; i++) {
      assertEquals(freq[i], input[i].getFrequency());
      assertEquals(memb[i], input[i] == map.get(input[i]));
    }
  }
  
 @Test
 public void testWritable() throws Exception {
   Gram one = new Gram("foo", 2, Gram.Type.HEAD);
   Gram two = new Gram("foobar", 3, Gram.Type.UNIGRAM);

   assertEquals("foo", one.getString());
   assertEquals(2, one.getFrequency());
   assertEquals(Gram.Type.HEAD, one.getType());

   assertEquals("foobar", two.getString());
   assertEquals(3, two.getFrequency());
   assertEquals(Gram.Type.UNIGRAM, two.getType());
   
   ByteArrayOutputStream bout = new ByteArrayOutputStream();
   DataOutput out = new DataOutputStream(bout);
   
   two.write(out);
   
   byte[] b = bout.toByteArray();
   
   ByteArrayInputStream bin = new ByteArrayInputStream(b);
   DataInput din = new DataInputStream(bin);
   
   one.readFields(din);

   assertEquals("foobar", one.getString());
   assertEquals(3, one.getFrequency());
   assertEquals(Gram.Type.UNIGRAM, one.getType());
   
 }
 
 @Test
 public void testSorting() {
   Gram[] input =
   {
    new Gram("foo", 2, Gram.Type.HEAD),
    new Gram("foo", 3, Gram.Type.HEAD),
    new Gram("foo", 4, Gram.Type.TAIL),
    new Gram("foo", 5, Gram.Type.TAIL),
    new Gram("bar", 6, Gram.Type.HEAD),
    new Gram("bar", 7, Gram.Type.TAIL),
    new Gram("bar", 8, Gram.Type.NGRAM),
    new Gram("bar", Gram.Type.UNIGRAM)
   };
   
   Gram[] sorted = new Gram[input.length];
   
   int[] expectations = {
       4, 0, 1, 5, 2, 3, 7, 6
   };
   
   
   System.arraycopy(input, 0, sorted, 0, input.length);
   
   Arrays.sort(sorted);
   
   for (int i=0; i < sorted.length; i++) {
     assertSame(input[expectations[i]], sorted[i]);
   }
 }
}
