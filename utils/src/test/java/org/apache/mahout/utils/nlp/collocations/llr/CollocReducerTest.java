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

import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.HEAD;
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.TAIL;
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.UNIGRAM;

import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

/**
 * Test the CollocReducer FIXME: add negative test cases.
 */
public class CollocReducerTest {
  
  OutputCollector<Gram,Gram> output;
  Reporter reporter;
  
  @Before
  @SuppressWarnings("unchecked")
  public void setUp() {
    output = EasyMock.createMock(OutputCollector.class);
    reporter = EasyMock.createMock(Reporter.class);
  }

  @Test
  public void testReduce() throws Exception {
    // test input, input[*][0] is the key,
    // input[*][1..n] are the values passed in via
    // the iterator.
    Gram[][] input = new Gram[][] {
        { new Gram("the",   UNIGRAM), new Gram("the", UNIGRAM), new Gram("the", UNIGRAM) },
        { new Gram("the",   HEAD), new Gram("the best"), new Gram("the worst") },
        { new Gram("of",    HEAD), new Gram("of times"), new Gram("of times") },
        { new Gram("times", TAIL), new Gram("of times"), new Gram("of times") }
    };

    // expected results.
    Gram[][] values = new Gram[][] {
        { new Gram("the", 4, UNIGRAM), new Gram("the", 2, UNIGRAM) },                             
        { new Gram("the best",  1), new Gram("the", 2,   HEAD) }, 
        { new Gram("the worst", 1), new Gram("the", 2,   HEAD) }, 
        { new Gram("of times",  2), new Gram("of",  2,   HEAD) }, 
        { new Gram("of times",  2), new Gram("times", 2, TAIL) }
    };

    // set up expectations
    for (Gram[] v : values) {
      output.collect(v[0], v[1]);
    }
    EasyMock.replay(reporter, output);
    
    // play back the input data.
    CollocReducer c = new CollocReducer();
    
    for (Gram[] ii : input) {
      List<Gram> vv = new LinkedList<Gram>();
      for (int i = 1; i < ii.length; i++) {
        vv.add(ii[i]);
      }
      c.reduce(ii[0], vv.iterator(), output, reporter);
    }
    
    EasyMock.verify(reporter, output);
  }
  
}
