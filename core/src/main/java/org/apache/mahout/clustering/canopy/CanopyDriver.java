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

package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;

import java.io.IOException;

public class CanopyDriver {

  private CanopyDriver() {
  }

  public static void main(String[] args) throws IOException {
    String input = args[0];
    String output = args[1];
    String measureClassName = args[2];
    double t1 = Double.parseDouble(args[3]);
    double t2 = Double.parseDouble(args[4]);
    runJob(input, output, measureClassName, t1, t2);
  }

  /**
   * Run the job
   *
   * @param input            the input pathname String
   * @param output           the output pathname String
   * @param measureClassName the DistanceMeasure class name
   * @param t1               the T1 distance threshold
   * @param t2               the T2 distance threshold
   */
  public static void runJob(String input, String output,
                            String measureClassName, double t1, double t2) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(
            org.apache.mahout.clustering.canopy.CanopyDriver.class);
    conf.set(Canopy.DISTANCE_MEASURE_KEY, measureClassName);
    conf.set(Canopy.T1_KEY, String.valueOf(t1));
    conf.set(Canopy.T2_KEY, String.valueOf(t2));

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(CanopyMapper.class);
    conf.setCombinerClass(CanopyCombiner.class);
    conf.setReducerClass(CanopyReducer.class);
    conf.setNumReduceTasks(1);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    client.setConf(conf);
    FileSystem dfs = FileSystem.get(conf);
    if (dfs.exists(outPath))
      dfs.delete(outPath, true);
    JobClient.runJob(conf);
  }

}
