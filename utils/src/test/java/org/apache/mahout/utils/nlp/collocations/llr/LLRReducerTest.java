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

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
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
@SuppressWarnings("deprecation")
public class LLRReducerTest {

  private static final Logger log = 
    LoggerFactory.getLogger(LLRReducerTest.class);

  Reporter reporter;
  LLCallback ll;
  LLCallback cl;
  // not verifying the llr algo output here, just the input, but it is handy
  // to see the values emitted.
  OutputCollector<Text,DoubleWritable> collector = new OutputCollector<Text,DoubleWritable>() {
    @Override
    public void collect(Text key, DoubleWritable value) throws IOException {
      log.info(key.toString() + " " + value.toString());
    }
  };


  @Before
  public void setUp() {
    reporter  = EasyMock.createMock(Reporter.class);
    ll        = EasyMock.createMock(LLCallback.class);
    cl        = new LLCallback() {
      @Override
      public double logLikelihoodRatio(int k11, int k12, int k21, int k22) {
        log.info("k11:" + k11 + " k12:" + k12 + " k21:" + k21 + " k22:" + k22);
        try {
          return LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22);
        }
        catch (Exception e) {
          e.printStackTrace();
          return -1;
        }
      }

    };
  }

  @Test
  public void testReduce() throws Exception {
    LLRReducer reducer = new LLRReducer(ll);

    // test input, input[*][0] is the key, 
    // input[*][1..n] are the values passed in via
    // the iterator.
    
    
    Gram[][] input = {
        {new Gram("the best",  1), new Gram("the",   2, HEAD), new Gram("best",  1, TAIL) },
        {new Gram("best of",   1), new Gram("best",  1, HEAD), new Gram("of",    2, TAIL) },
        {new Gram("of times",  2), new Gram("of",    2, HEAD), new Gram("times", 2, TAIL) },
        {new Gram("times the", 1), new Gram("times", 1, HEAD), new Gram("the",   1, TAIL) },
        {new Gram("the worst", 1), new Gram("the",   2, HEAD), new Gram("worst", 1, TAIL) },
        {new Gram("worst of",  1), new Gram("worst", 1, HEAD), new Gram("of",    2, TAIL) }
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

    JobConf config = new JobConf(CollocDriver.class);
    config.set(LLRReducer.NGRAM_TOTAL, "7");
    reducer.configure(config);

    for (Gram[] ii: input) {
      List<Gram> vv = new LinkedList<Gram>();
      for (int i = 1; i < ii.length; i++) {
        vv.add(ii[i]);
      }
      reducer.reduce(ii[0], vv.iterator(), collector, reporter);
    }

    EasyMock.verify(ll);
  }
}
