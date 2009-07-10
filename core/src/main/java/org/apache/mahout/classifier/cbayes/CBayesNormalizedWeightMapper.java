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

package org.apache.mahout.classifier.cbayes;

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

public class CBayesNormalizedWeightMapper extends MapReduceBase implements
    Mapper<Text, DoubleWritable, Text, DoubleWritable> {

  private static final Logger log = LoggerFactory.getLogger(CBayesNormalizedWeightMapper.class);

  private Map<String, Double> thetaNormalizer = null;

  /**
   * We need to calculate the idf of each feature in each label
   *
   * @param key The label,feature pair (can either be the freq Count or the term Document count
   */
  @Override
  public void map(Text key, DoubleWritable value,
                  OutputCollector<Text, DoubleWritable> output, Reporter reporter)
      throws IOException {

    String labelFeaturePair = key.toString();

    String label = labelFeaturePair.split(",")[0];
    output.collect(key, new DoubleWritable(-Math.log(value.get()) / thetaNormalizer.get(label)));// output -D_ij

  }

  @Override
  public void configure(JobConf job) {
    try {
      if (thetaNormalizer == null) {
        thetaNormalizer = new HashMap<String, Double>();

        DefaultStringifier<Map<String, Double>> mapStringifier = new DefaultStringifier<Map<String, Double>>(
            job, GenericsUtil.getClass(thetaNormalizer));

        String thetaNormalizationsString = mapStringifier.toString(thetaNormalizer);
        thetaNormalizationsString = job.get("cnaivebayes.thetaNormalizations",
            thetaNormalizationsString);
        thetaNormalizer = mapStringifier.fromString(thetaNormalizationsString);

      }
    } catch (IOException ex) {
      log.warn(ex.toString(), ex);
    }
  }
}
