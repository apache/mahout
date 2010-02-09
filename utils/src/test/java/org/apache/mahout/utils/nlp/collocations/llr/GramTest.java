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

import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Position.HEAD;
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Position.TAIL;

import java.util.HashMap;

import junit.framework.TestCase;

import org.apache.mahout.utils.nlp.collocations.llr.Gram;
import org.junit.Test;

public class GramTest {
  
  @Test
  public void testEquality() {
    Gram one = new Gram("foo", 2, HEAD);
    Gram two = new Gram("foo", 3, HEAD);
    
    TestCase.assertTrue(one.equals(two));
    TestCase.assertTrue(two.equals(one));
    
    Gram three = new Gram("foo", 4, TAIL);
    Gram four = new Gram("foo");
    
    TestCase.assertTrue(!three.equals(two));
    TestCase.assertTrue(four.equals(one));
    TestCase.assertTrue(one.equals(four));
    
    Gram five = new Gram("foobar", 4, TAIL);
    
    TestCase.assertTrue(!five.equals(four));
    TestCase.assertTrue(!five.equals(three));
    TestCase.assertTrue(!five.equals(two));
    TestCase.assertTrue(!five.equals(one));
  }
  
  @Test
  public void testHashing() {
    Gram[] input = 
    {
        new Gram("foo", 2, HEAD),
        new Gram("foo", 3, HEAD),
        new Gram("foo", 4, TAIL),
        new Gram("foo", 5, TAIL),
        new Gram("bar", 6, HEAD),
        new Gram("bar", 7, TAIL),
        new Gram("bar", 8),
        new Gram("bar")
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
        15,
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
        false,
        false
    };
    
    for (int i = 0; i < input.length; i++) {
      System.err.println(i);
      TestCase.assertEquals(freq[i], input[i].getFrequency());
      TestCase.assertEquals(memb[i], input[i] == map.get(input[i]));
    }
  }
}
