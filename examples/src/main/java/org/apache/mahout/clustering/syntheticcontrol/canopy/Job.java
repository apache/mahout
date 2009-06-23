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

package org.apache.mahout.clustering.syntheticcontrol.canopy;

import static org.apache.mahout.clustering.syntheticcontrol.Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.canopy.CanopyClusteringJob;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.SparseVector;

public class Job {
  private Job() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    if (args.length == 5) {
      String input = args[0];
      String output = args[1];
      String measureClassName = args[2];
      double t1 = Double.parseDouble(args[3]);
      double t2 = Double.parseDouble(args[4]);
      String vectorClassName = args[5];
      Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
      runJob(input, output, measureClassName, t1, t2, vectorClass);
    } else
      runJob("testdata", "output",
          "org.apache.mahout.utils.EuclideanDistanceMeasure", 80, 55, SparseVector.class);
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
      String measureClassName, double t1, double t2, Class<? extends Vector> vectorClass) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);

    Path outPath = new Path(output);
    client.setConf(conf);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath))
      dfs.delete(outPath, true);
    final String directoryContainingConvertedInput = output + DIRECTORY_CONTAINING_CONVERTED_INPUT;
    InputDriver.runJob(input, directoryContainingConvertedInput, vectorClass);
    CanopyClusteringJob.runJob(directoryContainingConvertedInput, output, measureClassName,
        t1, t2, vectorClass);
  }

}
