/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.meanshift;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;

public class MeanShiftCanopyDriver {

  public static void main(String[] args) {
    String input = args[0];
    String output = args[1];
    String measureClassName = args[2];
    double t1 = new Double(args[3]);
    double t2 = new Double(args[4]);
    double convergenceDelta = new Double(args[5]);
    runJob(input, output, measureClassName, t1, t2, convergenceDelta, false);
  }

  /**
   * Run the job
   * 
   * @param input the input pathname String
   * @param output the output pathname String
   * @param measureClassName the DistanceMeasure class name
   * @param t1 the T1 distance threshold
   * @param t2 the T2 distance threshold
   * @param convergenceDelta the double convergence criteria
   * @param inputIsSequenceFile true if input is sequence file encoded
   */
  public static void runJob(String input, String output,
      String measureClassName, double t1, double t2, double convergenceDelta,
      boolean inputIsSequenceFile) {

    JobClient client = new JobClient();
    JobConf conf = new JobConf(MeanShiftCanopyDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    conf.setInputPath(new Path(input));
    Path outPath = new Path(output);
    conf.setOutputPath(outPath);

    conf.setMapperClass(MeanShiftCanopyMapper.class);
    conf.setCombinerClass(MeanShiftCanopyCombiner.class);
    conf.setReducerClass(MeanShiftCanopyReducer.class);
    conf.setNumReduceTasks(1);
    if (inputIsSequenceFile)
      conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(MeanShiftCanopy.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(MeanShiftCanopy.CLUSTER_CONVERGENCE_KEY, "" + convergenceDelta);
    conf.set(MeanShiftCanopy.T1_KEY, "" + t1);
    conf.set(MeanShiftCanopy.T2_KEY, "" + t2);

    client.setConf(conf);
    try {
      JobClient.runJob(conf);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
