package org.apache.mahout.clustering.syntheticcontrol.canopy;

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
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.canopy.CanopyClusteringJob;

import java.io.IOException;

public class Job {

  public static void main(String[] args) throws Exception {
    if (args.length == 5) {
      String input = args[0];
      String output = args[1];
      String measureClassName = args[2];
      double t1 = Double.parseDouble(args[3]);
      double t2 = Double.parseDouble(args[4]);
      runJob(input, output, measureClassName, t1, t2);
    } else
      runJob("testdata", "output",
          "org.apache.mahout.utils.EuclideanDistanceMeasure", 80, 55);
  }

  /**
   * Run the canopy clustering job on an input dataset using the given distance
   * measure, t1 and t2 parameters. All output data will be written to the
   * output directory, which will be initially deleted if it exists. The
   * clustered points will reside in the path <output>/clustered-points. By
   * default, the job expects the a file containing synthetic_control.data as
   * obtained from
   * http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
   * resides in a directory named "testdata", and writes output to a directory
   * named "output".
   * 
   * @param input the String denoting the input directory path
   * @param output the String denoting the output directory path
   * @param measureClassName the String class name of the DistanceMeasure to use
   * @param t1 the canopy T1 threshold
   * @param t2 the canopy T2 threshold
   */
  private static void runJob(String input, String output,
      String measureClassName, double t1, double t2) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);

    Path outPath = new Path(output);
    client.setConf(conf);
    FileSystem dfs = FileSystem.get(conf);
    if (dfs.exists(outPath))
      dfs.delete(outPath, true);
    InputDriver.runJob(input, output + "/data");
    CanopyClusteringJob.runJob(output + "/data", output, measureClassName,
        t1, t2);
    OutputDriver.runJob(output + "/clusters", output + "/clustered-points");

  }

}
