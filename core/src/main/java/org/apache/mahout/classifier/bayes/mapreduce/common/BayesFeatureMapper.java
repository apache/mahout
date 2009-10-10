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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.classifier.BayesFileFormatter;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.nlp.NGrams;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Reads the input train set(preprocessed using the {@link BayesFileFormatter}). */
public class BayesFeatureMapper extends MapReduceBase implements
    Mapper<Text, Text, StringTuple, DoubleWritable> {

  private static final Logger log = LoggerFactory.getLogger(BayesFeatureMapper.class);

  private static final DoubleWritable one = new DoubleWritable(1.0);

  private int gramSize = 1;

  /**
   * We need to count the number of times we've seen a term with a given label and we need to output that. But this
   * Mapper does more than just outputing the count. It first does weight normalisation. Secondly, it outputs for each
   * unique word in a document value 1 for summing up as the Term Document Frequency. Which later is used to calculate
   * the Idf Thirdly, it outputs for each label the number of times a document was seen(Also used in Idf Calculation)
   *
   * @param key      The label
   * @param value    the features (all unique) associated w/ this label
   * @param output   The OutputCollector to write the results to
   * @param reporter Not used
   */
  @Override
  public void map(Text key, Text value,
                  OutputCollector<StringTuple, DoubleWritable> output, Reporter reporter)
      throws IOException {
    //String line = value.toString();
    String label = key.toString();

    Map<String, int[]> wordList = new HashMap<String, int[]>(1000);

    List<String> ngrams  = new NGrams(value.toString(), gramSize).generateNGramsWithoutLabel(); 

    for (String ngram : ngrams) {
      int[] count = wordList.get(ngram);
      if (count == null) {
        count = new int[1];
        count[0] = 0;
        wordList.put(ngram, count);
      }
      count[0]++;
    }
    double lengthNormalisation = 0.0;
    for (int[] D_kj : wordList.values()) {
      // key is label,word
      double dkjValue = (double) D_kj[0];
      lengthNormalisation += dkjValue * dkjValue;
    }
    lengthNormalisation = Math.sqrt(lengthNormalisation);

    // Output Length Normalized + TF Transformed Frequency per Word per Class
    // Log(1 + D_ij)/SQRT( SIGMA(k, D_kj) )
    for (Map.Entry<String, int[]> entry : wordList.entrySet()) {
      // key is label,word
      String token = entry.getKey();
      StringTuple tuple = new StringTuple();
      tuple.add(BayesConstants.WEIGHT);
      tuple.add(label);
      tuple.add(token);
      DoubleWritable f = new DoubleWritable(Math.log(1.0 + entry.getValue()[0]) / lengthNormalisation);
      output.collect(tuple, f);
    }
    reporter.setStatus("Bayes Feature Mapper: Document Label: " + label);  
    
    // Output Document Frequency per Word per Class
    
    for (String token : wordList.keySet()) {
      // key is label,word
      
      StringTuple dfTuple = new StringTuple();
      dfTuple.add(BayesConstants.DOCUMENT_FREQUENCY);
      dfTuple.add(label);
      dfTuple.add(token);      
      output.collect(dfTuple, one);
      
      StringTuple tokenCountTuple = new StringTuple();
      tokenCountTuple.add(BayesConstants.FEATURE_COUNT);
      tokenCountTuple.add(token);
      output.collect(tokenCountTuple, one);

    }

    // output that we have seen the label to calculate the Count of Document per
    // class
    StringTuple labelCountTuple = new StringTuple();
    labelCountTuple.add(BayesConstants.LABEL_COUNT);
    labelCountTuple.add(label);
    output.collect(labelCountTuple, one);
  }

  @Override
  public void configure(JobConf job) {
    try {
      System.out.println("Bayes Parameter" + job.get("bayes.parameters"));
      Parameters params = Parameters.fromString(job.get("bayes.parameters",""));
      gramSize = Integer.valueOf(params.get("gramSize"));

    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
}
