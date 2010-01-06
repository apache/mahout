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

package org.apache.mahout.classifier.bayes.mapreduce.cbayes;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mapper for Calculating the ThetaNormalizer for a label in Naive Bayes
 * Algorithm
 * 
 */
public class CBayesThetaNormalizerMapper extends MapReduceBase implements
    Mapper<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory
      .getLogger(CBayesThetaNormalizerMapper.class);
  
  private Map<String,Double> labelWeightSum;
  private double sigmaJSigmaK;
  private double vocabCount;
  private double alphaI = 1.0;
  
  /**
   * We need to calculate the idf of each feature in each label
   * 
   * @param key
   *          The label,feature pair (can either be the freq Count or the term
   *          Document count
   */
  @Override
  public void map(StringTuple key,
                  DoubleWritable value,
                  OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    
    if (key.stringAt(0).equals(BayesConstants.FEATURE_SUM)) { // if it is from
      // the Sigma_j
      // folder
      
      for (Map.Entry<String,Double> stringDoubleEntry : labelWeightSum
          .entrySet()) {
        String label = stringDoubleEntry.getKey();
        double weight = Math
            .log((value.get() + alphaI)
                 / (sigmaJSigmaK - stringDoubleEntry.getValue() + vocabCount));
        
        reporter.setStatus("Complementary Bayes Theta Normalizer Mapper: "
                           + stringDoubleEntry + " => " + weight);
        StringTuple normalizerTuple = new StringTuple(
            BayesConstants.LABEL_THETA_NORMALIZER);
        normalizerTuple.add(label);
        output.collect(normalizerTuple, new DoubleWritable(weight)); // output
        // Sigma_j
        
      }
      
    } else {
      String label = key.stringAt(1);
      
      double dIJ = value.get();
      double denominator = 0.5 * ((sigmaJSigmaK / vocabCount) + (dIJ * this.labelWeightSum
          .size()));
      double weight = Math.log(1.0 - dIJ / denominator);
      
      reporter.setStatus("Complementary Bayes Theta Normalizer Mapper: "
                         + label + " => " + weight);
      
      StringTuple normalizerTuple = new StringTuple(
          BayesConstants.LABEL_THETA_NORMALIZER);
      normalizerTuple.add(label);
      
      // output -D_ij
      output.collect(normalizerTuple, new DoubleWritable(weight));
      
    }
    
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      if (labelWeightSum == null) {
        labelWeightSum = new HashMap<String,Double>();
        
        DefaultStringifier<Map<String,Double>> mapStringifier = new DefaultStringifier<Map<String,Double>>(
            job, GenericsUtil.getClass(labelWeightSum));
        
        String labelWeightSumString = mapStringifier.toString(labelWeightSum);
        labelWeightSumString = job.get("cnaivebayes.sigma_k",
          labelWeightSumString);
        labelWeightSum = mapStringifier.fromString(labelWeightSumString);
        
        DefaultStringifier<Double> stringifier = new DefaultStringifier<Double>(
            job, GenericsUtil.getClass(sigmaJSigmaK));
        String sigmaJSigmaKString = stringifier.toString(sigmaJSigmaK);
        sigmaJSigmaKString = job.get("cnaivebayes.sigma_jSigma_k",
          sigmaJSigmaKString);
        sigmaJSigmaK = stringifier.fromString(sigmaJSigmaKString);
        
        String vocabCountString = stringifier.toString(vocabCount);
        vocabCountString = job.get("cnaivebayes.vocabCount", vocabCountString);
        vocabCount = stringifier.fromString(vocabCountString);
        
        Parameters params = Parameters.fromString(job.get("bayes.parameters",
          ""));
        alphaI = Double.valueOf(params.get("alpha_i", "1.0"));
        
      }
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
  
}
