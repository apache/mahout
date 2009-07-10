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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;

public class BayesWeightSummerMapper extends MapReduceBase implements
    Mapper<Text, DoubleWritable, Text, DoubleWritable> {

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
    int i = labelFeaturePair.indexOf(',');

    String label = labelFeaturePair.substring(0, i);
    String feature = labelFeaturePair.substring(i + 1);

    output.collect(new Text(',' + feature), value);//sum of weight for all labels for a feature Sigma_j
    output.collect(new Text('_' + label), value);//sum of weight for all features for a label Sigma_k
    output.collect(new Text("*"), value);//sum of weight of all features for all label Sigma_kSigma_j

  }
}
