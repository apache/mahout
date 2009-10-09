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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesFeatureDriver;
import org.apache.mahout.classifier.bayes.common.BayesParameters;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesJob;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesTfIdfDriver;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesWeightSummerDriver;
import org.apache.mahout.classifier.bayes.mapreduce.common.JobExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/** Create and run the Bayes Trainer. */
public class CBayesDriver implements BayesJob{

  private static final Logger log = LoggerFactory.getLogger(CBayesDriver.class);

  /**
   * Takes in two arguments: <ol> <li>The input {@link Path} where the input documents live</li>
   * <li>The output {@link Path} where to write the Model as a
   * {@link org.apache.hadoop.io.SequenceFile}</li> </ol>
   *
   * @param args The args input and output path.
   * @throws Exception in case of problems during job execution. 
   */
  public static void main(String[] args) throws Exception {
    JobExecutor executor = new JobExecutor();
    executor.execute(args, new CBayesDriver());
  }

  /**
   * Run the job
   *
   * @param input  the input pathname String
   * @param output the output pathname String
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  @Override
  public void runJob(String input, String output, BayesParameters params) throws IOException, InterruptedException, ClassNotFoundException {
    JobConf conf = new JobConf(CBayesDriver.class);
    Path outPath = new Path(output);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    log.info("Reading features...");
    //Read the features in each document normalized by length of each document
    BayesFeatureDriver feature = new BayesFeatureDriver();
    feature.runJob(input, output, params);

    log.info("Calculating Tf-Idf...");
    //Calculate the TfIdf for each word in each label
    BayesTfIdfDriver tfidf = new BayesTfIdfDriver();
    tfidf.runJob(input, output,params);

    log.info("Calculating weight sums for labels and features...");
    //Calculate the Sums of weights for each label, for each feature and for each feature and for each label
    BayesWeightSummerDriver summer = new BayesWeightSummerDriver();
    summer.runJob(input, output, params);

    //Calculate the W_ij = log(Theta) for each label, feature. This step actually generates the complement class
    //CBayesThetaDriver.runJob(input, output);

    log.info("Calculating the weight Normalisation factor for each complement class...");
    //Calculate the normalization factor Sigma_W_ij for each complement class.
    CBayesThetaNormalizerDriver normalizer = new CBayesThetaNormalizerDriver();
    normalizer.runJob(input, output, params);

    //Calculate the normalization factor Sigma_W_ij for each complement class.
    //CBayesNormalizedWeightDriver.runJob(input, output);

    Path docCountOutPath = new Path(output + "/trainer-docCount");
    if (dfs.exists(docCountOutPath)) {
      dfs.delete(docCountOutPath, true);
    }
    Path termDocCountOutPath = new Path(output + "/trainer-termDocCount");
    if (dfs.exists(termDocCountOutPath)) {
      dfs.delete(termDocCountOutPath, true);
    }
    Path featureCountOutPath = new Path(output + "/trainer-featureCount");
    if (dfs.exists(featureCountOutPath)) {
      dfs.delete(featureCountOutPath, true);
    }
    Path wordFreqOutPath = new Path(output + "/trainer-wordFreq");
    if (dfs.exists(wordFreqOutPath)) {
      dfs.delete(wordFreqOutPath, true);
    }
    Path vocabCountPath = new Path(output + "/trainer-tfIdf/trainer-vocabCount");
    if (dfs.exists(vocabCountPath)) {
      dfs.delete(vocabCountPath, true);
    }
    /*Path tfIdfOutPath = new Path(output+ "/trainer-tfIdf");
    if (dfs.exists(tfIdfOutPath))
      dfs.delete(tfIdfOutPath, true);*/
    Path vocabCountOutPath = new Path(output + "/trainer-vocabCount");
    if (dfs.exists(vocabCountOutPath)) {
      dfs.delete(vocabCountOutPath, true);
    }
    /* Path weightsOutPath = new Path(output+ "/trainer-weights");
 if (dfs.exists(weightsOutPath))
   dfs.delete(weightsOutPath, true);*/
    /*Path thetaOutPath = new Path(output+ "/trainer-theta");
    if (dfs.exists(thetaOutPath))
      dfs.delete(thetaOutPath, true);*/
    /*Path thetaNormalizerOutPath = new Path(output+ "/trainer-thetaNormalizer");
    if (dfs.exists(thetaNormalizerOutPath))
      dfs.delete(thetaNormalizerOutPath, true);*/

  }
}
