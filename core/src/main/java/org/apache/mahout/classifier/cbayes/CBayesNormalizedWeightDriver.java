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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

/** Create and run the Bayes Trainer. */
public class CBayesNormalizedWeightDriver {

  private static final Logger log = LoggerFactory.getLogger(CBayesNormalizedWeightDriver.class);

  private CBayesNormalizedWeightDriver() {
  }

  /**
   * Takes in two arguments: <ol> <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the {@link org.apache.mahout.common.Model} as a
   * {@link org.apache.hadoop.io.SequenceFile}</li> </ol>
   *
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
   * @param input  the input pathname String
   * @param output the output pathname String
   */
  public static void runJob(String input, String output) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(CBayesNormalizedWeightDriver.class);


    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DoubleWritable.class);
    FileInputFormat.addInputPath(conf, new Path(output + "/trainer-theta"));
    Path outPath = new Path(output + "/trainer-weight");
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setNumMapTasks(100);
    //conf.setNumReduceTasks(1);
    conf.setMapperClass(CBayesNormalizedWeightMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(CBayesNormalizedWeightReducer.class);
    conf.setReducerClass(CBayesNormalizedWeightReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.set("io.serializations",
        "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    // Dont ever forget this. People should keep track of how hadoop conf parameters and make or break a piece of code
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    Path thetaNormalizationsFiles = new Path(output + "/trainer-thetaNormalizer/part*");
    Map<String, Double> thetaNormalizer = SequenceFileModelReader.readLabelSums(dfs, thetaNormalizationsFiles, conf);
    double perLabelWeightSumNormalisationFactor = Double.MAX_VALUE;
    for (Map.Entry<String, Double> stringDoubleEntry1 : thetaNormalizer.entrySet()) {

      double Sigma_W_ij = stringDoubleEntry1.getValue();
      if (perLabelWeightSumNormalisationFactor > Math.abs(Sigma_W_ij)) {
        perLabelWeightSumNormalisationFactor = Math.abs(Sigma_W_ij);
      }
    }

    for (Map.Entry<String, Double> stringDoubleEntry : thetaNormalizer.entrySet()) {
      double Sigma_W_ij = stringDoubleEntry.getValue();
      thetaNormalizer.put(stringDoubleEntry.getKey(), Sigma_W_ij / perLabelWeightSumNormalisationFactor);
    }


    DefaultStringifier<Map<String, Double>> mapStringifier =
        new DefaultStringifier<Map<String, Double>>(conf, GenericsUtil.getClass(thetaNormalizer));
    String thetaNormalizationsString = mapStringifier.toString(thetaNormalizer);

    Map<String, Double> c = mapStringifier.fromString(thetaNormalizationsString);
    log.info("{}", c);
    conf.set("cnaivebayes.thetaNormalizations", thetaNormalizationsString);


    client.setConf(conf);

    JobClient.runJob(conf);

  }
}
