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

package org.apache.mahout.classifier.bayes.mapreduce.common;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Can also be used as a local Combiner beacuse only two values should be there inside the values
 */
public class BayesTfIdfReducer extends MapReduceBase implements
    Reducer<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesTfIdfReducer.class);
  
  @Override
  public void reduce(StringTuple key,
                     Iterator<DoubleWritable> values,
                     OutputCollector<StringTuple,DoubleWritable> output,
                     Reporter reporter) throws IOException {
    // Key is label,word, value is the number of times we've seen this label word per local node. Output is the same
    
    if (key.stringAt(0).equals(BayesConstants.FEATURE_SET_SIZE)) {
      double vocabCount = 0.0;
      
      while (values.hasNext()) {
        reporter.setStatus("Bayes TfIdf Reducer: vocabCount " + vocabCount);
        vocabCount += values.next().get();
      }
      
      log.info("{}\t{}", key, vocabCount);
      output.collect(key, new DoubleWritable(vocabCount));
    } else if (key.stringAt(0).equals(BayesConstants.WEIGHT)) {
      double idfTimesDIJ = 1.0;
      while (values.hasNext()) {
        idfTimesDIJ *= values.next().get();
      }
      reporter.setStatus("Bayes TfIdf Reducer: " + key + " => " + idfTimesDIJ);
      output.collect(key, new DoubleWritable(idfTimesDIJ));
    } else {
      throw new IllegalArgumentException("Unexpected StringTuple: " + key);
    }
  }

}
