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
import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.map.OpenObjectDoubleHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive Bayes Tfidf Mapper. Calculates per document statistics
 * 
 */
public class BayesTfIdfMapper extends MapReduceBase implements
    Mapper<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesTfIdfMapper.class);
  
  private static final StringTuple VOCAB_COUNT = new StringTuple(BayesConstants.FEATURE_SET_SIZE);
  
  private static final DoubleWritable ONE = new DoubleWritable(1.0);
  
  private final OpenObjectDoubleHashMap<String> labelDocumentCounts = new OpenObjectDoubleHashMap<String>();
  
  /**
   * We need to calculate the Tf-Idf of each feature in each label
   * 
   * @param key
   *          The label,feature pair (can either be the freq Count or the term Document count
   */
  @Override
  public void map(StringTuple key,
                  DoubleWritable value,
                  OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    
    if (key.length() == 3) {
      if (key.stringAt(0).equals(BayesConstants.WEIGHT)) {
        reporter.setStatus("Bayes TfIdf Mapper: Tf: " + key);
        output.collect(key, value);
      } else if (key.stringAt(0).equals(BayesConstants.DOCUMENT_FREQUENCY)) {
        String label = key.stringAt(1);
        Double labelDocumentCount = labelDocumentCounts.get(label);
        double logIdf = Math.log(labelDocumentCount / value.get());
        key.replaceAt(0, BayesConstants.WEIGHT);
        output.collect(key, new DoubleWritable(logIdf));
        reporter.setStatus("Bayes TfIdf Mapper: log(Idf): " + key);
      } else {
        throw new IllegalArgumentException("Unrecognized Tuple: " + key);
      }
    } else if (key.length() == 2) {
      if (key.stringAt(0).equals(BayesConstants.FEATURE_COUNT)) {
        output.collect(VOCAB_COUNT, ONE);
        reporter.setStatus("Bayes TfIdf Mapper: vocabCount");
      } else {
        throw new IllegalArgumentException("Unexpected Tuple: " + key);
      }
    }
    
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      this.labelDocumentCounts.clear();
      Map<String,Double> labelDocCountTemp = Maps.newHashMap();
      
      DefaultStringifier<Map<String,Double>> mapStringifier = new DefaultStringifier<Map<String,Double>>(job,
          GenericsUtil.getClass(labelDocCountTemp));
      
      String labelDocumentCountString =
          job.get("cnaivebayes.labelDocumentCounts", mapStringifier.toString(labelDocCountTemp));
      
      labelDocCountTemp = mapStringifier.fromString(labelDocumentCountString);
      for (Map.Entry<String, Double> stringDoubleEntry : labelDocCountTemp.entrySet()) {
        this.labelDocumentCounts.put(stringDoubleEntry.getKey(), stringDoubleEntry.getValue());
      }
      
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
  
}
