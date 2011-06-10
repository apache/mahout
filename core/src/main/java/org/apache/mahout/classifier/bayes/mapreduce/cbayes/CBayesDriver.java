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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureDriver;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesJob;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesTfIdfDriver;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesWeightSummerDriver;
import org.apache.mahout.common.HadoopUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Create and run the Bayes Trainer. */
public class CBayesDriver implements BayesJob {
  
  private static final Logger log = LoggerFactory.getLogger(CBayesDriver.class);

  @Override
  public void runJob(Path input, Path output, BayesParameters params) throws IOException {
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, output);
    
    log.info("Reading features...");
    // Read the features in each document normalized by length of each document
    BayesFeatureDriver feature = new BayesFeatureDriver();
    feature.runJob(input, output, params);
    
    log.info("Calculating Tf-Idf...");
    // Calculate the TfIdf for each word in each label
    BayesTfIdfDriver tfidf = new BayesTfIdfDriver();
    tfidf.runJob(input, output, params);
    
    log.info("Calculating weight sums for labels and features...");
    // Calculate the Sums of weights for each label, for each feature and for
    // each feature and for each label
    BayesWeightSummerDriver summer = new BayesWeightSummerDriver();
    summer.runJob(input, output, params);
    
    log.info("Calculating the weight Normalisation factor for each complement class...");
    // Calculate the normalization factor Sigma_W_ij for each complement class.
    CBayesThetaNormalizerDriver normalizer = new CBayesThetaNormalizerDriver();
    normalizer.runJob(input, output, params);
    
    if (params.isSkipCleanup()) {
      return;
    }
    
    Path docCountOutPath = new Path(output, "trainer-docCount");
    HadoopUtil.delete(conf, docCountOutPath);

    Path termDocCountOutPath = new Path(output, "trainer-termDocCount");
    HadoopUtil.delete(conf, termDocCountOutPath);

    Path featureCountOutPath = new Path(output, "trainer-featureCount");
    HadoopUtil.delete(conf, featureCountOutPath);

    Path wordFreqOutPath = new Path(output, "trainer-wordFreq");
    HadoopUtil.delete(conf, wordFreqOutPath);

    Path vocabCountPath = new Path(output, "trainer-tfIdf/trainer-vocabCount");
    HadoopUtil.delete(conf, vocabCountPath);

    Path vocabCountOutPath = new Path(output, "trainer-vocabCount");
    HadoopUtil.delete(conf, vocabCountOutPath);

    
  }
}
