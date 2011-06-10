/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.classifier.bayes;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureMapper;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureReducer;
import org.apache.mahout.classifier.bayes.mapreduce.common.FeatureLabelComparator;
import org.apache.mahout.common.DummyOutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.StringTuple;
import org.junit.Test;

public final class BayesFeatureMapReduceTest extends MahoutTestCase {

  private static DummyOutputCollector<StringTuple,DoubleWritable> runMapReduce(BayesParameters bp) throws IOException {
    
    BayesFeatureMapper mapper = new BayesFeatureMapper();
    JobConf conf = new JobConf();
    conf.set("io.serializations",
      "org.apache.hadoop.io.serializer.JavaSerialization,"
          + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    conf.set("bayes.parameters", bp.toString());
    mapper.configure(conf);
    
    DummyOutputCollector<StringTuple,DoubleWritable> mapperOutput = new DummyOutputCollector<StringTuple,DoubleWritable>();
    
    mapper.map(new Text("foo"), new Text("big brown shoe"), mapperOutput, Reporter.NULL);
    mapper.map(new Text("foo"), new Text("cool chuck taylors"), mapperOutput, Reporter.NULL);
    
    mapper.map(new Text("bar"), new Text("big big dog"), mapperOutput, Reporter.NULL);
    mapper.map(new Text("bar"), new Text("cool rain"), mapperOutput, Reporter.NULL);
   
    mapper.map(new Text("baz"), new Text("red giant"), mapperOutput, Reporter.NULL);
    mapper.map(new Text("baz"), new Text("white dwarf"), mapperOutput, Reporter.NULL);
    mapper.map(new Text("baz"), new Text("cool black hole"), mapperOutput, Reporter.NULL);
    
    BayesFeatureReducer reducer = new BayesFeatureReducer();
    reducer.configure(conf);
    
    DummyOutputCollector<StringTuple,DoubleWritable> reducerOutput = new DummyOutputCollector<StringTuple,DoubleWritable>();
    Map<StringTuple, List<DoubleWritable>> outputData = mapperOutput.getData();
    
    // put the mapper output in the expected order (emulate shuffle)
    FeatureLabelComparator cmp = new FeatureLabelComparator();
    Collection<StringTuple> keySet = new TreeSet<StringTuple>(cmp);
    keySet.addAll(mapperOutput.getKeys());
    
    for (StringTuple k: keySet) {
      List<DoubleWritable> v = outputData.get(k);
      reducer.reduce(k, v.iterator(), reducerOutput, Reporter.NULL);
    }

    return reducerOutput;
  }

  @Test
  public void testNoFilters() throws Exception {
    BayesParameters bp = new BayesParameters();
    bp.setGramSize(1);
    bp.setMinDF(1);
    DummyOutputCollector<StringTuple,DoubleWritable> reduceOutput = runMapReduce(bp);

    assertCounts(reduceOutput, 
        17, /* df: 13 unique term/label pairs */
        14, /* fc: 12 unique features across all labels */
        3,  /* lc: 3 labels */
        17  /* wt: 13 unique term/label pairs */);
  }

  @Test
  public void testMinSupport() throws Exception {
    BayesParameters bp = new BayesParameters();
    bp.setGramSize(1);
    bp.setMinSupport(2);
    DummyOutputCollector<StringTuple,DoubleWritable> reduceOutput = runMapReduce(bp);
    
    assertCounts(reduceOutput, 
        5, /* df: 5 unique term/label pairs */
        2, /* fc: 'big' and 'cool' appears more than 2 times */
        3, /* lc: 3 labels */
        5  /* wt: 5 unique term/label pairs */);
    
  }

  @Test
  public void testMinDf() throws Exception {
    BayesParameters bp = new BayesParameters();
    bp.setGramSize(1);
    bp.setMinDF(2);
    DummyOutputCollector<StringTuple,DoubleWritable> reduceOutput = runMapReduce(bp);
    
    // 13 unique term/label pairs. 3 labels
    // should be a df and fc for each pair, no filtering
    assertCounts(reduceOutput, 
        5, /* df: 5 term/label pairs contains terms in more than 2 document */
        2, /* fc */
        3,  /* lc */
        5  /* wt */);
    
  }

  @Test
  public void testMinBoth() throws Exception {
    BayesParameters bp = new BayesParameters();
    bp.setGramSize(1);
    bp.setMinSupport(3);
    bp.setMinDF(2);
    DummyOutputCollector<StringTuple,DoubleWritable> reduceOutput = runMapReduce(bp);
    
    // 13 unique term/label pairs. 3 labels
    // should be a df and fc for each pair, no filtering
    assertCounts(reduceOutput, 
        5, /* df: 5 term/label pairs contains terms in more than 2 document */
        2, /* fc: 'cool' appears 3 times */
        3,  /* lc */
        5  /* wt */);
  }
  
  private static void assertCounts(DummyOutputCollector<StringTuple,DoubleWritable> output,
                                   int dfExpected,
                                   int fcExpected,
                                   int lcExpected,
                                   int wtExpected) {
    int dfCount = 0;
    int fcCount = 0;
    int lcCount = 0;
    int wtCount = 0;
    
    Map<StringTuple, List<DoubleWritable>> outputData = output.getData();
    for (Map.Entry<StringTuple, List<DoubleWritable>> entry: outputData.entrySet()) {
      String type = entry.getKey().stringAt(0);
      if (type.equals(BayesConstants.DOCUMENT_FREQUENCY)) {
        dfCount++;
      } else if (type.equals(BayesConstants.FEATURE_COUNT)) {
        fcCount++;
      } else if (type.equals(BayesConstants.LABEL_COUNT)) {
        lcCount++;
      } else if (type.equals(BayesConstants.WEIGHT)) {
        wtCount++;
      }
      assertEquals("value size", 1, entry.getValue().size());
    }
    
    assertEquals("document frequency count", dfExpected, dfCount);
    assertEquals("feature count", fcExpected, fcCount);
    assertEquals("label count", lcExpected, lcCount);
    assertEquals("feature weight count", wtExpected, wtCount);
  }
}
