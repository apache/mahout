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

import static org.apache.mahout.vectorizer.collocations.llr.Gram.Type.HEAD;
import static org.apache.mahout.vectorizer.collocations.llr.Gram.Type.NGRAM;
import static org.apache.mahout.vectorizer.collocations.llr.Gram.Type.TAIL;
import static org.apache.mahout.vectorizer.collocations.llr.Gram.Type.UNIGRAM;

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.common.MahoutTestCase;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

/**
 * Test the CollocReducer
 */
public final class CollocReducerTest extends MahoutTestCase {
  
  private Reducer<GramKey,Gram,Gram,Gram>.Context context;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    context = EasyMock.createMock(Context.class);
  }
  
  @Test
  public void testReduce() throws Exception {
    // test input, input[*][0] is the key,
    // input[*][1..n] are the values passed in via
    // the iterator.
    Gram[][] input = {
        {new Gram("the", UNIGRAM), new Gram("the", UNIGRAM), new Gram("the", UNIGRAM)},
        {new Gram("the", HEAD), new Gram("the best", NGRAM), new Gram("the worst", NGRAM)},
        {new Gram("of", HEAD), new Gram("of times", NGRAM), new Gram("of times", NGRAM)},
        {new Gram("times", TAIL), new Gram("of times", NGRAM), new Gram("of times", NGRAM)}};
    
    // expected results.
    Gram[][] values = {{new Gram("the", 2, UNIGRAM), new Gram("the", 2, UNIGRAM)},
                                    {new Gram("the best", 1, NGRAM), new Gram("the", 2, HEAD)},
                                    {new Gram("the worst", 1, NGRAM), new Gram("the", 2, HEAD)},
                                    {new Gram("of times", 2, NGRAM), new Gram("of", 2, HEAD)},
                                    {new Gram("of times", 2, NGRAM), new Gram("times", 2, TAIL)}};

    // set up expectations
    for (Gram[] v : values) {
      context.write(v[0], v[1]);
    }
    EasyMock.replay(context);
    
    // play back the input data.
    CollocReducer c = new CollocReducer();
    
    GramKey key = new GramKey();

    byte[] empty = new byte[0];
    for (Gram[] ii : input) {
      key.set(ii[0], empty);

      Collection<Gram> vv = new LinkedList<Gram>();
      vv.addAll(Arrays.asList(ii));
      c.reduce(key, vv, context);
    }
    
    EasyMock.verify(context);
  }
  
}
