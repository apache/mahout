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
import static org.apache.mahout.utils.nlp.collocations.llr.NGramCollector.Count.NGRAM_TOTAL;

import java.io.Reader;
import java.io.StringReader;
import java.util.Collections;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.utils.nlp.collocations.llr.Gram.Type;
import org.easymock.EasyMock;
import org.junit.Before;
import org.junit.Test;

/** Test for NGramCollectorTest
 * FIXME: Add negative test cases
 */
@SuppressWarnings("deprecation")
public class NGramCollectorTest {
  
  OutputCollector<Gram,Gram> collector;
  Reporter reporter;
  
  @Before
  @SuppressWarnings("unchecked")
  public void setUp() {
    collector = EasyMock.createMock(OutputCollector.class);
    reporter  = EasyMock.createMock(Reporter.class);
  }
  
  @Test
  public void testCollectNgrams() throws Exception {
    
    String input = "the best of times the worst of times";
    
    String[][] values =
      new String[][]{
                     {"h_the",   "the best"},
                     {"t_best",  "the best"},
                     {"h_best",  "best of"},
                     {"t_of",    "best of"},
                     {"h_of",    "of times"},
                     {"t_times", "of times"},
                     {"h_times", "times the"},
                     {"t_the",   "times the"},
                     {"h_the",   "the worst"},
                     {"t_worst", "the worst"},
                     {"h_worst", "worst of"},
                     {"t_of",    "worst of"},
                     {"h_of",    "of times"},
                     {"t_times", "of times"}
    };
    // set up expectations for mocks. ngram max size = 2
    
    // setup expectations
    for (String[] v: values) {
      Type p = v[0].startsWith("h") ? HEAD : TAIL;
      Gram subgram = new Gram(v[0].substring(2), p);
      Gram ngram = new Gram(v[1]);
      collector.collect(subgram, ngram);
    }
    
    reporter.incrCounter(NGRAM_TOTAL, 7);
    EasyMock.replay(reporter, collector);
    
    Reader r = new StringReader(input);
    
    JobConf conf = new JobConf();
    conf.set(NGramCollector.MAX_SHINGLE_SIZE, "2");
    conf.set(NGramCollector.ANALYZER_CLASS, TestAnalyzer.class.getName());
    
    NGramCollector c = new NGramCollector();
    c.configure(conf);
    
    c.collectNgrams(r, collector, reporter);
    
    EasyMock.verify(reporter, collector);
  }
  
  /** A lucene 2.9 standard analyzer with no stopwords. */
  public static class TestAnalyzer extends Analyzer {
    final Analyzer a;
    
    public TestAnalyzer() {
      a = new StandardAnalyzer(Version.LUCENE_29, Collections.EMPTY_SET);
    }
    
    @Override
    public TokenStream tokenStream(String arg0, Reader arg1) {
      return a.tokenStream(arg0, arg1);
    }
  }
}
