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

import org.apache.mahout.classifier.bayes.BayesParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.StringTuple;

import com.google.common.base.Preconditions;

/** Can also be used as a local Combiner. A simple summing reducer */
public class BayesFeatureReducer extends MapReduceBase implements
    Reducer<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesFeatureReducer.class);
  
  private double minSupport = -1.0;
  private double minDf      = -1.0;
  
  private String currentDfFeature;
  private double currentCorpusDf;
  private double currentCorpusTf;
  
  @Override
  public void reduce(StringTuple key,
                     Iterator<DoubleWritable> values,
                     OutputCollector<StringTuple,DoubleWritable> output,
                     Reporter reporter) throws IOException {
    
    // StringTuple key is either:
    // type, word        for type=FEATURE_COUNT, FEATURE_TF or WEIGHT tuples
    // type, label       for type=LABEL_COUNT_TUPLES
    // type, label, word for type=DOCUMENT_FREQUENCY tuples
    
    double sum = 0.0;
    while (values.hasNext()) {  
      reporter.setStatus("Feature Reducer:" + key);
      sum += values.next().get();
    }
    reporter.setStatus("Bayes Feature Reducer: " + key + " => " + sum);

    Preconditions.checkArgument(key.length() >= 2 && key.length() <= 3,
                                "StringTuple length out of bounds, not (2 < length < 3)");
    
    int featureIndex = key.length() == 2 ? 1 : 2;
    
    // FeatureLabelComparator guarantees that for a given label, we will
    // see FEATURE_TF items first, FEATURE_COUNT items second, 
    // DOCUMENT_FREQUENCY items next and finally WEIGHT items, while
    // the FeaturePartitioner guarantees that all tuples containing a given term
    // will be handled by the same reducer.
    if (key.stringAt(0).equals(BayesConstants.LABEL_COUNT)) {
      // no-op, just collect
      output.collect(key, new DoubleWritable(sum));

    } else if (key.stringAt(0).equals(BayesConstants.FEATURE_TF)) {
      currentDfFeature = key.stringAt(1);
      currentCorpusTf = sum;
      currentCorpusDf = -1;
      
      if (minSupport > 0.0 && currentCorpusTf < minSupport) {
        reporter.incrCounter("skipped", "less_than_minSupport", 1);
      }
      // never emit FEATURE_TF tuples.

    } else if (!key.stringAt(featureIndex).equals(currentDfFeature)) {
      throw new IllegalStateException("Found feature data " + key + " prior to feature tf");

    } else if (minSupport > 0.0 && currentCorpusTf < minSupport) {
      reporter.incrCounter("skipped", "less_than_minSupport_label-term", 1);
      // skip items that have less than a specified frequency.

    } else if (key.stringAt(0).equals(BayesConstants.FEATURE_COUNT)) {
      currentCorpusDf = sum;
      if (minDf > 0.0 && currentCorpusDf < minDf) {
        reporter.incrCounter("skipped", "less_than_minDf", 1);
        // skip items that have less than the specified minSupport.
      } else {
        output.collect(key, new DoubleWritable(sum));
      }

    } else if (currentCorpusDf == -1.0) {
      throw new IllegalStateException("Found feature data " + key + " prior to feature count");

    } else if (minDf > 0.0 && currentCorpusDf < minDf) {
      reporter.incrCounter("skipped", "less_than_minDf_label-term", 1);
      // skip items that have less than a specified frequency.

    } else {
      output.collect(key, new DoubleWritable(sum));
    }
  }

  @Override
  public void configure(JobConf job) {
    try {
      BayesParameters params = new BayesParameters(job.get("bayes.parameters", ""));
      log.info("Bayes Parameter {}", params.print());
      minSupport = params.getMinSupport();
      minDf      = params.getMinDF();
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
}
