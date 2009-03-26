/**
 * Licensed to the Apache Software Foundation (ASF) under one
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

package org.apache.mahout.clustering.kmeans;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public class KMeansJob {

  private KMeansJob() {
  }

  public static void main(String[] args) throws IOException {

    if (args.length != 7) {
      System.err.println("Expected number of arguments 10 and received:"
          + args.length);
      System.err
          .println("Usage:input clustersIn output measureClass convergenceDelta maxIterations numCentroids");
      throw new IllegalArgumentException();
    }
    int index = 0;
    String input = args[index++];
    String clusters = args[index++];
    String output = args[index++];
    String measureClass = args[index++];
    double convergenceDelta = Double.parseDouble(args[index++]);
    int maxIterations = Integer.parseInt(args[index++]);
    int numCentroids = Integer.parseInt(args[index++]);

    runJob(input, clusters, output, measureClass, convergenceDelta,
        maxIterations, numCentroids);
  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it
   * exists beforehand
   * 
   * @param input the directory pathname for input points
   * @param clustersIn the directory pathname for initial & computed clusters
   * @param output the directory pathname for output points
   * @param measureClass the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param maxIterations the maximum number of iterations
   */
  public static void runJob(String input, String clustersIn, String output,
      String measureClass, double convergenceDelta, int maxIterations,
      int numCentroids) throws IOException {
    // delete the output directory
    JobConf conf = new JobConf(KMeansJob.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(conf);
    if (fs.exists(outPath)) {
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);

    KMeansDriver.runJob(input, clustersIn, output, measureClass,
        convergenceDelta, maxIterations, numCentroids);
  }
}
