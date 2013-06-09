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
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;

import com.google.common.io.Closeables;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class GramKeyTest extends MahoutTestCase {

  @Test
  public void testGramKeySort() {
    byte[] foo = {1};
    //byte[] bar = new byte[1]; bar[0] = 2;
    
    
    // byte argument in GramKey breaks tie between equal grams
    byte[] empty = new byte[0];
    GramKey[] input = {
      new GramKey(new Gram("bar", 1, Gram.Type.UNIGRAM), empty),
      new GramKey(new Gram("bar", 1, Gram.Type.UNIGRAM), empty),
      new GramKey(new Gram("bar", 1, Gram.Type.UNIGRAM), foo),
      new GramKey(new Gram("bar", 8, Gram.Type.NGRAM), foo),
      new GramKey(new Gram("bar", 8, Gram.Type.NGRAM), empty),
      new GramKey(new Gram("foo", 2, Gram.Type.HEAD), foo),
      new GramKey(new Gram("foo", 3, Gram.Type.HEAD), empty),
      new GramKey(new Gram("foo", 4, Gram.Type.TAIL), foo),
      new GramKey(new Gram("foo", 5, Gram.Type.TAIL), foo),
      new GramKey(new Gram("bar", 6, Gram.Type.HEAD), foo),
      new GramKey(new Gram("bar", 7, Gram.Type.TAIL), empty),
    };
    
    int[] expect = {
        9, 6, 5, 10, 7, 8, 0, 1, 2, 4, 3
    };
    
    GramKey[] sorted = new GramKey[input.length];
    
    System.arraycopy(input, 0, sorted, 0, input.length);
    
    Arrays.sort(sorted);

    for (int i=0; i < input.length; i++) {
      assertSame(input[expect[i]], sorted[i]);
    }
  }
  
  @Test
  public void testWritable() throws Exception {
    byte[] foo = new byte[0];
    byte[] bar = {2};

    GramKey one = new GramKey(new Gram("foo", 2, Gram.Type.HEAD), foo);
    GramKey two = new GramKey(new Gram("foobar", 3, Gram.Type.UNIGRAM), bar);

    assertEquals("foo", one.getPrimaryString());
    assertEquals("foobar", two.getPrimaryString());
    
    assertEquals(Gram.Type.UNIGRAM, two.getType());
    
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(bout);

    try {
      two.write(out);
    } finally {
      Closeables.close(out, false);
    }
    
    byte[] b = bout.toByteArray();
    
    ByteArrayInputStream bin = new ByteArrayInputStream(b);
    DataInputStream din = new DataInputStream(bin);

    try {
      one.readFields(din);
    } finally {
      Closeables.close(din, true);
    }

    assertTrue(Arrays.equals(two.getBytes(), one.getBytes()));
    assertEquals(Gram.Type.UNIGRAM, one.getType());
    
  }
}
