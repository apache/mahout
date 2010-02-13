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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.StringTuple;

/**
 * 
 * Calculates weight sum for a unique label, and feature
 * 
 */
public class BayesWeightSummerMapper extends MapReduceBase implements
    Mapper<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  /**
   * We need to calculate the weight sums across each label and each feature
   * 
   * @param key
   *          The label,feature tuple containing the tfidf value
   */
  @Override
  public void map(StringTuple key,
                  DoubleWritable value,
                  OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    String label = key.stringAt(1);
    String feature = key.stringAt(2);
    reporter.setStatus("Bayes Weight Summer Mapper: " + key);
    StringTuple featureSum = new StringTuple(BayesConstants.FEATURE_SUM);
    featureSum.add(feature);
    output.collect(featureSum, value); // sum of weight for all labels for a
    // feature Sigma_j
    StringTuple labelSum = new StringTuple(BayesConstants.LABEL_SUM);
    labelSum.add(label);
    output.collect(labelSum, value); // sum of weight for all features for a
    // label Sigma_k
    StringTuple totalSum = new StringTuple(BayesConstants.TOTAL_SUM);
    output.collect(totalSum, value); // sum of weight of all features for all
    // label Sigma_kSigma_j
    
  }
}
