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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

public class GramTest {
  
  @Test
  public void testConstructorsGetters() {
    Gram one = new Gram("foo", 2, Gram.Type.HEAD);
    
    Assert.assertEquals("foo", one.getString());
    Assert.assertEquals(2, one.getFrequency());
    Assert.assertEquals(Gram.Type.HEAD, one.getType());
    
    Gram oneClone = new Gram(one);
    
    Assert.assertEquals("foo", oneClone.getString());
    Assert.assertEquals(2, oneClone.getFrequency());
    Assert.assertEquals(Gram.Type.HEAD, oneClone.getType());
    
    Gram two = new Gram("foo", 3, Gram.Type.TAIL);
    Assert.assertEquals(Gram.Type.TAIL, two.getType());
    
    Gram three = new Gram("foo", 4, Gram.Type.UNIGRAM);
    Assert.assertEquals(Gram.Type.UNIGRAM, three.getType());
    
    Gram four = new Gram("foo", 5, Gram.Type.NGRAM);
    Assert.assertEquals(Gram.Type.NGRAM, four.getType());
   
    try {
      new Gram(null, 4, Gram.Type.UNIGRAM);
      Assert.fail("expected exception");
    } catch (NullPointerException ex) {
      /* ok */
    }
   
    
    try {
      new Gram("foo", 4, null);
      Assert.fail("expected exception");
    } catch (NullPointerException ex) {
      /* ok */
    }
  }
  
  @Test
  public void testEquality() {
    Gram one = new Gram("foo", 2, Gram.Type.HEAD);  
    Gram two = new Gram("foo", 3, Gram.Type.HEAD);

    Assert.assertEquals(one, two);
    Assert.assertEquals(two, one);
    
    Gram three = new Gram("foo", 4, Gram.Type.TAIL);
    Gram four = new Gram("foo", Gram.Type.UNIGRAM);
    
    Assert.assertTrue(!three.equals(two));
    Assert.assertTrue(!four.equals(one));
    Assert.assertTrue(!one.equals(four));
    
    Gram five = new Gram("foo", 5, Gram.Type.UNIGRAM);

    Assert.assertEquals(four, five);
    
    Gram six = new Gram("foo", 6, Gram.Type.NGRAM);
    Gram seven = new Gram("foo", 7, Gram.Type.NGRAM);
    
    Assert.assertTrue(!five.equals(six));
    Assert.assertEquals(six, seven);
    
    Gram eight = new Gram("foobar", 4, Gram.Type.TAIL);
    
    Assert.assertTrue(!eight.equals(four));
    Assert.assertTrue(!eight.equals(three));
    Assert.assertTrue(!eight.equals(two));
    Assert.assertTrue(!eight.equals(one));
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
    
    HashMap<Gram,Gram> map = new HashMap<Gram,Gram>();
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
      Assert.assertEquals(freq[i], input[i].getFrequency());
      Assert.assertEquals(memb[i], input[i] == map.get(input[i]));
    }
  }
  
 @Test
 public void testWritable() throws IOException {
   Gram one = new Gram("foo", 2, Gram.Type.HEAD);
   Gram two = new Gram("foobar", 3, Gram.Type.UNIGRAM);

   Assert.assertEquals("foo", one.getString());
   Assert.assertEquals(2, one.getFrequency());
   Assert.assertEquals(Gram.Type.HEAD, one.getType());

   Assert.assertEquals("foobar", two.getString());
   Assert.assertEquals(3, two.getFrequency());
   Assert.assertEquals(Gram.Type.UNIGRAM, two.getType());
   
   ByteArrayOutputStream bout = new ByteArrayOutputStream();
   DataOutput out = new DataOutputStream(bout);
   
   two.write(out);
   
   byte[] b = bout.toByteArray();
   
   ByteArrayInputStream bin = new ByteArrayInputStream(b);
   DataInput din = new DataInputStream(bin);
   
   one.readFields(din);

   Assert.assertEquals("foobar", one.getString());
   Assert.assertEquals(3, one.getFrequency());
   Assert.assertEquals(Gram.Type.UNIGRAM, one.getType());
   
 }
 
 @Test
 public void testSorting() throws IOException {
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
     Assert.assertSame(input[expectations[i]], sorted[i]);
   }
 }
}
