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

import java.io.IOException;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringTuple;

/** Create and run the Bayes Trainer. */
public class BayesWeightSummerDriver implements BayesJob {

  @Override
  public void runJob(Path input, Path output, BayesParameters params) throws IOException {
    Configurable client = new JobClient();
    JobConf conf = new JobConf(BayesWeightSummerDriver.class);
    conf.setJobName("Bayes Weight Summer Driver running over input: " + input);
    
    conf.setOutputKeyClass(StringTuple.class);
    conf.setOutputValueClass(DoubleWritable.class);
    
    FileInputFormat.addInputPath(conf, new Path(output, "trainer-tfIdf/trainer-tfIdf"));
    Path outPath = new Path(output, "trainer-weights");
    FileOutputFormat.setOutputPath(conf, outPath);
    HadoopUtil.delete(conf, outPath);
    // conf.setNumReduceTasks(1);
    // conf.setNumMapTasks(100);
    conf.setMapperClass(BayesWeightSummerMapper.class);
    // see the javadoc for the spec for file input formats: first token is key,
    // rest is input. Whole document on one line
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(BayesWeightSummerReducer.class);
    conf.setReducerClass(BayesWeightSummerReducer.class);
    conf.setOutputFormat(BayesWeightSummerOutputFormat.class);

    conf.set("bayes.parameters", params.toString());
    conf.set("output.table", output.toString());
    
    client.setConf(conf);
    
    JobClient.runJob(conf);
  }
}
