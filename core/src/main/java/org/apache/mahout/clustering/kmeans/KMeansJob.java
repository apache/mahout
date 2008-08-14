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
package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

public class KMeansJob {

  private KMeansJob() {
  }

  public static void main(String[] args) {
    String input = args[0];
    String clusters = args[1];
    String output = args[2];
    String measureClass = args[3];
    double convergenceDelta = Double.parseDouble(args[4]);
    int maxIterations = Integer.parseInt(args[5]);
    runJob(input, clusters, output, measureClass, convergenceDelta,
        maxIterations);
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
      String measureClass, double convergenceDelta, int maxIterations) {
    try {
      // delete the output directory
      JobConf conf = new JobConf(KMeansJob.class);
      Path outPath = new Path(output);
      FileSystem fs = FileSystem.get(conf);
      if (fs.exists(outPath)) {
        fs.delete(outPath, true);
      }
      fs.mkdirs(outPath);
      KMeansDriver.runJob(input, clustersIn, output, measureClass,
          convergenceDelta, maxIterations);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
