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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;

import java.io.IOException;

/** Create and run the Bayes Trainer. */
public class BayesWeightSummerDriver {
  private BayesWeightSummerDriver() {
  }

  /**
   * Takes in two arguments: <ol> <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the the interim files as a {@link
   * org.apache.hadoop.io.SequenceFile}</li> </ol>
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
    JobConf conf = new JobConf(BayesWeightSummerDriver.class);


    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.addInputPath(conf, new Path(output + "/trainer-tfIdf/trainer-tfIdf"));
    Path outPath = new Path(output + "/trainer-weights");
    FileOutputFormat.setOutputPath(conf, outPath);
    //conf.setNumReduceTasks(1);
    conf.setNumMapTasks(100);
    conf.setMapperClass(BayesWeightSummerMapper.class);
    //see the javadoc for the spec for file input formats: first token is key, rest is input.  Whole document on one line
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(BayesWeightSummerReducer.class);
    conf.setReducerClass(BayesWeightSummerReducer.class);
    conf.setOutputFormat(BayesWeightSummerOutputFormat.class);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    client.setConf(conf);

    JobClient.runJob(conf);
  }
}
