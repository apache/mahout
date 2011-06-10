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

package org.apache.mahout.classifier.bayes.mapreduce.bayes;

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
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.map.OpenObjectDoubleHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mapper for Calculating the ThetaNormalizer for a label in Naive Bayes Algorithm
 * 
 */
public class BayesThetaNormalizerMapper extends MapReduceBase implements
    Mapper<StringTuple,DoubleWritable,StringTuple,DoubleWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(BayesThetaNormalizerMapper.class);
  
  private final OpenObjectDoubleHashMap<String> labelWeightSum = new OpenObjectDoubleHashMap<String>();
  private double sigmaJSigmaK;
  private double vocabCount;
  private double alphaI = 1.0;
  
  /**
   * We need to calculate the thetaNormalization factor of each label
   * 
   * @param key
   *          The label,feature pair
   * @param value
   *          The tfIdf of the pair
   */
  @Override
  public void map(StringTuple key,
                  DoubleWritable value,
                  OutputCollector<StringTuple,DoubleWritable> output,
                  Reporter reporter) throws IOException {
    
    String label = key.stringAt(1);
    
    reporter.setStatus("Bayes Theta Normalizer Mapper: " + label);
    
    double weight = Math.log((value.get() + alphaI) / (labelWeightSum.get(label) + vocabCount));
    StringTuple thetaNormalizerTuple = new StringTuple(BayesConstants.LABEL_THETA_NORMALIZER);
    thetaNormalizerTuple.add(label);
    output.collect(thetaNormalizerTuple, new DoubleWritable(weight));
  }
  
  @Override
  public void configure(JobConf job) {
    try {
      labelWeightSum.clear();
      Map<String,Double> labelWeightSumTemp = Maps.newHashMap();
      
      DefaultStringifier<Map<String,Double>> mapStringifier = new DefaultStringifier<Map<String,Double>>(job,
          GenericsUtil.getClass(labelWeightSumTemp));
      
      String labelWeightSumString = job.get("cnaivebayes.sigma_k", mapStringifier.toString(labelWeightSumTemp));
      labelWeightSumTemp = mapStringifier.fromString(labelWeightSumString);
      for (Map.Entry<String, Double> stringDoubleEntry : labelWeightSumTemp.entrySet()) {
        this.labelWeightSum.put(stringDoubleEntry.getKey(), stringDoubleEntry.getValue());
      }
      DefaultStringifier<Double> stringifier = new DefaultStringifier<Double>(job, GenericsUtil
          .getClass(sigmaJSigmaK));
      String sigmaJSigmaKString = job.get("cnaivebayes.sigma_jSigma_k", stringifier.toString(sigmaJSigmaK));
      sigmaJSigmaK = stringifier.fromString(sigmaJSigmaKString);
      
      String vocabCountString = stringifier.toString(vocabCount);
      vocabCountString = job.get("cnaivebayes.vocabCount", vocabCountString);
      vocabCount = stringifier.fromString(vocabCountString);
      
      Parameters params = new Parameters(job.get("bayes.parameters", ""));
      alphaI = Double.valueOf(params.get("alpha_i", "1.0"));
      
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
  
}
