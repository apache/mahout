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
import org.apache.hadoop.util.GenericsUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class BayesTfIdfMapper extends MapReduceBase implements
        Mapper<Text, DoubleWritable, Text, DoubleWritable> {

  private static final Logger log = LoggerFactory.getLogger(BayesTfIdfMapper.class);

  private Map<String, Double> labelDocumentCounts = null;

  /**
   * We need to calculate the Tf-Idf of each feature in each label
   *
   * @param key The label,feature pair (can either be the freq Count or the term Document count
   */
  @Override
  public void map(Text key, DoubleWritable value,
                  OutputCollector<Text, DoubleWritable> output, Reporter reporter)
          throws IOException {

    String labelFeaturePair = key.toString();

    char firstChar = labelFeaturePair.charAt(0);
    switch (firstChar) {
      case '-': {// if it is the termDocumentCount
        labelFeaturePair = labelFeaturePair.substring(1);
        //-17th_century_mathematicians_anderson__alexander,1582
        int idx = labelFeaturePair.indexOf(",");
        if (idx != -1) {
          String label = labelFeaturePair.substring(0, idx);

          Double labelDocumentCount = labelDocumentCounts.get(label);
          if (labelDocumentCount == null) {
            throw new IOException("Invalid label: " + label);
          }
          double logIdf = Math.log(labelDocumentCount / value.get());
          output.collect(new Text(labelFeaturePair), new DoubleWritable(logIdf));
        } else {
          throw new IOException("Invalid ");
        }
        break;
      }
      case ',': {
        output.collect(new Text("*vocabCount"), new DoubleWritable(1.0));
        break;
      }
      default: {
        output.collect(key, value);
        break;
      }
    }
  }

  @Override
  public void configure(JobConf job) {
    try {
      if (labelDocumentCounts == null) {
        labelDocumentCounts = new HashMap<String, Double>();

        DefaultStringifier<Map<String, Double>> mapStringifier =
                new DefaultStringifier<Map<String, Double>>(job, GenericsUtil.getClass(labelDocumentCounts));

        String labelDocumentCountString = mapStringifier.toString(labelDocumentCounts);
        labelDocumentCountString = job.get("cnaivebayes.labelDocumentCounts", labelDocumentCountString);

        labelDocumentCounts = mapStringifier.fromString(labelDocumentCountString);
      }
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
}
