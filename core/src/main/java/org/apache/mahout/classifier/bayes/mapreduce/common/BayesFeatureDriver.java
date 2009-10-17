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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.KeyValueTextInputFormat;
import org.apache.mahout.classifier.bayes.common.BayesParameters;
import org.apache.mahout.common.StringTuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/** Create and run the Bayes Feature Reader Step. */
public class BayesFeatureDriver implements BayesJob {

  private static final Logger log = LoggerFactory.getLogger(BayesFeatureDriver.class);

  /**
   * Run the job
   *
   * @param input  the input pathname String
   * @param output the output pathname String
   */
  @Override
  public void runJob(String input, String output, BayesParameters params) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(BayesFeatureDriver.class);
    conf.setJobName("Bayes Feature Driver running over input: " +  input);
    conf.setOutputKeyClass(StringTuple.class);
    conf.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);
    
    conf.setMapperClass(BayesFeatureMapper.class);

    conf.setInputFormat(KeyValueTextInputFormat.class);
    conf.setCombinerClass(BayesFeatureReducer.class);
    conf.setReducerClass(BayesFeatureReducer.class);
    conf.setOutputFormat(BayesFeatureOutputFormat.class);
    conf.set("io.serializations",
        "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    // this conf parameter needs to be set enable serialisation of conf values

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    conf.set("bayes.parameters", params.toString());

    client.setConf(conf);
    JobClient.runJob(conf);

  }
}
