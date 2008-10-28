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

package org.apache.mahout.classifier.bayes.common;

import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.classifier.BayesFileFormatter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Reads the input train set(preprocessed using the {@link BayesFileFormatter}).
 */
public class BayesFeatureMapper extends MapReduceBase implements
    Mapper<Text, Text, Text, DoubleWritable> {

  private static final Logger log = LoggerFactory.getLogger(BayesFeatureMapper.class);

  private static final DoubleWritable one = new DoubleWritable(1.0);

  private final Text labelWord = new Text();

  private int gramSize = 1;

  /**
   * We need to count the number of times we've seen a term with a given label
   * and we need to output that. But this Mapper does more than just outputing the count. It first does weight
   * normalisation.
   * Secondly, it outputs for each unique word in a document value 1 for summing up as the Term Document Frequency.
   * Which later is used to calculate the Idf
   * Thirdly, it outputs for each label the number of times a document was seen(Also used in Idf Calculation)
   * 
   * @param key The label
   * @param value the features (all unique) associated w/ this label
   * @param output
   * @param reporter
   * @throws IOException
   */
  public void map(Text key, Text value,
      OutputCollector<Text, DoubleWritable> output, Reporter reporter)
      throws IOException {
    //String line = value.toString();
    String label = key.toString();
    int keyLen = label.length();

    Map<String, Integer> wordList = new HashMap<String, Integer>(1000);
    // TODO: srowen wonders where wordList is ever updated?

    StringBuilder builder = new StringBuilder(label);
    builder.ensureCapacity(32);// make sure we have a reasonably size buffer to
                               // begin with
    //List<String> previousN_1Grams  = Model.generateNGramsWithoutLabel(line, keyLen);
    
    double lengthNormalisation = 0.0;
    for (double D_kj : wordList.values()) {
      // key is label,word
      lengthNormalisation += D_kj * D_kj;
    }
    lengthNormalisation = Math.sqrt(lengthNormalisation);

    // Ouput Length Normalized + TF Transformed Frequency per Word per Class
    // Log(1 + D_ij)/SQRT( SIGMA(k, D_kj) )
    for (Map.Entry<String, Integer> entry : wordList.entrySet()) {
      // key is label,word
      String token = entry.getKey();
      builder.append(',').append(token);
      labelWord.set(builder.toString());
      DoubleWritable f = new DoubleWritable(Math.log(1.0 + entry.getValue()) / lengthNormalisation);
      output.collect(labelWord, f);
      builder.setLength(keyLen);// truncate back
    }

    // Ouput Document Frequency per Word per Class
    String dflabel = "-" + label;
    int dfKeyLen = dflabel.length();
    builder = new StringBuilder(dflabel);
    for (String token : wordList.keySet()) {
      // key is label,word
      builder.append(',').append(token);
      labelWord.set(builder.toString());
      output.collect(labelWord, one);
      output.collect(new Text("," + token), one);
      builder.setLength(dfKeyLen);// truncate back

    }

    // ouput that we have seen the label to calculate the Count of Document per
    // class
    output.collect(new Text("_" + label), one);
  }

  @Override
  public void configure(JobConf job) {
    try {

      DefaultStringifier<Integer> intStringifier = new DefaultStringifier<Integer>(job, Integer.class);

      String gramSizeString = intStringifier.toString(gramSize);
      gramSizeString = job.get("bayes.gramSize", gramSizeString);
      gramSize = intStringifier.fromString(gramSizeString);

    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
}
