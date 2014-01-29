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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.vectorizer.collocations.llr.Gram.Type;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

/**
 * Test for CollocMapper 
 */
public final class CollocMapperTest extends MahoutTestCase {
  
  private Mapper<Text,StringTuple,GramKey,Gram>.Context context;
  private Counter counter;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    counter = EasyMock.createMock(Counter.class);
    context = EasyMock.createMock(Context.class);
  }
  
  @Test
  public void testCollectNgrams() throws Exception {
    
    Text key = new Text();
    key.set("dummy-key");
    
    String[] input = {"the", "best", "of", "times", "the", "worst", "of",
    "times"};
    StringTuple inputTuple = new StringTuple();
    for (String i : input) {
      inputTuple.add(i);
    }
    
    String[][] values = { {"h_the", "the best"},
                          {"t_best", "the best"},
                          {"h_of", "of times"},
                          {"t_times", "of times"},
                          {"h_best", "best of"},
                          {"t_of", "best of"},
                          {"h_the", "the worst"},
                          {"t_worst", "the worst"},
                          {"h_times", "times the"},
                          {"t_the", "times the"},
                          {"h_worst", "worst of"},
                          {"t_of", "worst of"},};
    // set up expectations for mocks. ngram max size = 2
    
    Configuration conf = getConfiguration();
    conf.set(CollocMapper.MAX_SHINGLE_SIZE, "2");
    EasyMock.expect(context.getConfiguration()).andReturn(conf);
    
    for (String[] v : values) {
      Type p = v[0].startsWith("h") ? Gram.Type.HEAD : Gram.Type.TAIL;
      int frequency = 1;
      if ("of times".equals(v[1])) {
        frequency = 2;
      }
      
      Gram subgram = new Gram(v[0].substring(2), frequency, p);
      Gram ngram = new Gram(v[1], frequency, Gram.Type.NGRAM);
      
      GramKey subgramKey = new GramKey(subgram, new byte[0]);
      GramKey subgramNgramKey = new GramKey(subgram, ngram.getBytes());

      context.write(subgramKey, subgram);
      context.write(subgramNgramKey, ngram);
    }
    EasyMock.expect(context.getCounter(CollocMapper.Count.NGRAM_TOTAL)).andReturn(counter);
    counter.increment(7);
    EasyMock.replay(context,counter);

    CollocMapper c = new CollocMapper();
    c.setup(context);
    
    c.map(key, inputTuple, context);
    
    EasyMock.verify(context);
  }
  
  @Test
  public void testCollectNgramsWithUnigrams() throws Exception {
    
    Text key = new Text();
    key.set("dummy-key");
    
    String[] input = {"the", "best", "of", "times", "the", "worst", "of",
    "times"};
    StringTuple inputTuple = new StringTuple();
    for (String i : input) {
      inputTuple.add(i);
    }
    
    String[][] values = {{"h_the", "the best"},
                                         {"t_best", "the best"},
                                         {"h_of", "of times"},
                                         {"t_times", "of times"},
                                         {"h_best", "best of"},
                                         {"t_of", "best of"},
                                         {"h_the", "the worst"},
                                         {"t_worst", "the worst"},
                                         {"h_times", "times the"},
                                         {"t_the", "times the"},
                                         {"h_worst", "worst of"},
                                         {"t_of", "worst of"},
                                         {"u_worst", "worst"}, {"u_of", "of"},
                                         {"u_the", "the"}, {"u_best", "best"},
                                         {"u_times", "times"},};

    // set up expectations for mocks. ngram max size = 2
    Configuration conf = getConfiguration();
    conf.set(CollocMapper.MAX_SHINGLE_SIZE, "2");
    conf.setBoolean(CollocDriver.EMIT_UNIGRAMS, true);
    EasyMock.expect(context.getConfiguration()).andReturn(conf);
    
    for (String[] v : values) {
      Type p = v[0].startsWith("h") ? Gram.Type.HEAD : Gram.Type.TAIL;
      p = v[0].startsWith("u") ? Gram.Type.UNIGRAM : p;
      int frequency = 1;
      if ("of times".equals(v[1]) || "of".equals(v[1]) || "times".equals(v[1])
          || "the".equals(v[1])) {
        frequency = 2;
      }
      
      
     
      if (p == Gram.Type.UNIGRAM) {
        Gram unigram = new Gram(v[1], frequency, Gram.Type.UNIGRAM);
        GramKey unigramKey = new GramKey(unigram, new byte[0]);
        context.write(unigramKey, unigram);
      }
      else {
        Gram subgram = new Gram(v[0].substring(2), frequency, p);
        Gram ngram = new Gram(v[1], frequency, Gram.Type.NGRAM);
        
        GramKey subgramKey = new GramKey(subgram, new byte[0]);
        GramKey subgramNgramKey = new GramKey(subgram, ngram.getBytes());
        context.write(subgramKey, subgram);
        context.write(subgramNgramKey, ngram);
      }
    }
    
    EasyMock.expect(context.getCounter(CollocMapper.Count.NGRAM_TOTAL)).andReturn(counter);
    counter.increment(7);
    EasyMock.replay(context,counter);
    
    CollocMapper c = new CollocMapper();
    c.setup(context);
    
    c.map(key, inputTuple, context);
    
    EasyMock.verify(context);
  }
}
