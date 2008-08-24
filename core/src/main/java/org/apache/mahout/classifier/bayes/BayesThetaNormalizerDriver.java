package org.apache.mahout.classifier.bayes;
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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.io.IOException;

/**
 * Create and run the Bayes Theta Normalization Step.
 */
public class BayesThetaNormalizerDriver {

  private static final Logger log = LoggerFactory.getLogger(BayesThetaNormalizerDriver.class);

  /**
   * Takes in two arguments:
   * <ol>
   * <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the the interim filesas a {@link org.apache.hadoop.io.SequenceFile}</li>
   * </ol>
   * @param args The args
   */
  public static void main(String[] args) throws IOException {
    String input = args[0];
    String output = args[1];

    runJob(input, output);
  }

  /**
   * Run the job
   *
   * @param input            the input pathname String
   * @param output           the output pathname String
   */
  public static void runJob(String input, String output) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(BayesThetaNormalizerDriver.class);
    

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(FloatWritable.class);
    SequenceFileInputFormat.addInputPath(conf, new Path(output + "/trainer-tfIdf/trainer-tfIdf"));
    Path outPath = new Path(output + "/trainer-thetaNormalizer");
    SequenceFileOutputFormat.setOutputPath(conf, outPath);
    conf.setNumMapTasks(100);
    //conf.setNumReduceTasks(1);
    conf.setMapperClass(BayesThetaNormalizerMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(BayesThetaNormalizerReducer.class);    
    conf.setReducerClass(BayesThetaNormalizerReducer.class);    
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization"); // Dont ever forget this. People should keep track of how hadoop conf parameters and make or break a piece of code
    
    FileSystem dfs = FileSystem.get(conf);
    if (dfs.exists(outPath))
      dfs.delete(outPath, true);

    SequenceFileModelReader reader = new SequenceFileModelReader();

    Path Sigma_kFiles = new Path(output+"/trainer-weights/Sigma_k/*");
    HashMap<String,Float> labelWeightSum= reader.readLabelSums(dfs, Sigma_kFiles, conf);
    DefaultStringifier<HashMap<String,Float>> mapStringifier = new DefaultStringifier<HashMap<String,Float>>(conf, GenericsUtil.getClass(labelWeightSum));
    String labelWeightSumString = mapStringifier.toString(labelWeightSum);

    log.info("Sigma_k for Each Label");
    Map<String,Float> c = mapStringifier.fromString(labelWeightSumString);
    log.info("{}", c);
    conf.set("cnaivebayes.sigma_k", labelWeightSumString);


    Path sigma_kSigma_jFile = new Path(output+"/trainer-weights/Sigma_kSigma_j/*");
    Float sigma_jSigma_k = reader.readSigma_jSigma_k(dfs, sigma_kSigma_jFile, conf);
    DefaultStringifier<Float> floatStringifier = new DefaultStringifier<Float>(conf, Float.class);
    String sigma_jSigma_kString = floatStringifier.toString(sigma_jSigma_k);

    log.info("Sigma_kSigma_j for each Label and for each Features");
    Float retSigma_jSigma_k = floatStringifier.fromString(sigma_jSigma_kString);
    log.info("{}", retSigma_jSigma_k);
    conf.set("cnaivebayes.sigma_jSigma_k", sigma_jSigma_kString);

    Path vocabCountFile = new Path(output+"/trainer-tfIdf/trainer-vocabCount/*");
    Float vocabCount = reader.readVocabCount(dfs, vocabCountFile, conf);
    String vocabCountString = floatStringifier.toString(vocabCount);

    log.info("Vocabulary Count");
    conf.set("cnaivebayes.vocabCount", vocabCountString);
    Float retvocabCount = floatStringifier.fromString(vocabCountString);
    log.info("{}", retvocabCount);

    client.setConf(conf);

    JobClient.runJob(conf);
    
  }
}
