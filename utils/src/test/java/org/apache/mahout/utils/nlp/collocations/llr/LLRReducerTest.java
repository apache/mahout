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
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.NGRAM;
import static org.apache.mahout.utils.nlp.collocations.llr.Gram.Type.TAIL;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.utils.nlp.collocations.llr.LLRReducer.LLCallback;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Test the LLRReducer
 *  FIXME: Add negative test cases.
 */
public class LLRReducerTest {
  
  private static final Logger log =
    LoggerFactory.getLogger(LLRReducerTest.class);
  
  private LLCallback ll;
  private LLCallback cl;

  @Before
  public void setUp() {
    ll        = EasyMock.createMock(LLCallback.class);
    cl        = new LLCallback() {
      @Override
      public double logLikelihoodRatio(int k11, int k12, int k21, int k22) {
        log.info("k11:{} k12:{} k21:{} k22:{}", new Object[] {k11, k12, k21, k22});
        return LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22);
      }
    };
  }
  
  @Test
  public void testReduce() throws Exception {
    
    // test input, input[*][0] is the key,
    // input[*][1..n] are the values passed in via
    // the iterator.
    
    
    Gram[][] input = {
                      {new Gram("the best",  1, NGRAM), new Gram("the",   2, HEAD), new Gram("best",  1, TAIL) },
                      {new Gram("best of",   1, NGRAM), new Gram("best",  1, HEAD), new Gram("of",    2, TAIL) },
                      {new Gram("of times",  2, NGRAM), new Gram("of",    2, HEAD), new Gram("times", 2, TAIL) },
                      {new Gram("times the", 1, NGRAM), new Gram("times", 1, HEAD), new Gram("the",   1, TAIL) },
                      {new Gram("the worst", 1, NGRAM), new Gram("the",   2, HEAD), new Gram("worst", 1, TAIL) },
                      {new Gram("worst of",  1, NGRAM), new Gram("worst", 1, HEAD), new Gram("of",    2, TAIL) }
    };
    
    int[][] expectations = {
                            // A+B, A+!B, !A+B, !A+!B
                            {1, 1, 0, 5}, // the best
                            {1, 0, 1, 5}, // best of
                            {2, 0, 0, 5}, // of times
                            {1, 0, 0, 6}, // times the
                            {1, 1, 0, 5}, // the worst
                            {1, 0, 1, 5}  // worst of
    };
    
    for (int[] ee: expectations) {
      EasyMock.expect(ll.logLikelihoodRatio(ee[0], ee[1], ee[2], ee[3])).andDelegateTo(cl);
    }
    
    EasyMock.replay(ll);
    
    Configuration conf = new Configuration();
    conf.set(LLRReducer.NGRAM_TOTAL, "7");
    LLRReducer reducer = new LLRReducer(ll);
    DummyRecordWriter<Text, DoubleWritable> writer = new DummyRecordWriter<Text, DoubleWritable>();
    Reducer<Gram, Gram, Text, DoubleWritable>.Context context = DummyRecordWriter.build(reducer,
                                                                                          conf,
                                                                                          writer,
                                                                                          Gram.class,
                                                                                          Gram.class);
    reducer.setup(context);
    
    for (Gram[] ii: input) {
      List<Gram> vv = new LinkedList<Gram>();
      vv.addAll(Arrays.asList(ii).subList(1, ii.length));
      reducer.reduce(ii[0], vv, context);
    }
    
    EasyMock.verify(ll);
  }
}
