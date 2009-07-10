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

package org.apache.mahout.clustering.meanshift;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class MeanShiftCanopyJob {

  protected static final String CONTROL_CONVERGED = "/control/converged";

  private static final Logger log = LoggerFactory
      .getLogger(MeanShiftCanopyJob.class);

  private MeanShiftCanopyJob() {
  }

  public static void main(String[] args) throws IOException {
    String input = args[0];
    String output = args[1];
    String measureClassName = args[2];
    double t1 = Double.parseDouble(args[3]);
    double t2 = Double.parseDouble(args[4]);
    double convergenceDelta = Double.parseDouble(args[5]);
    int maxIterations = Integer.parseInt(args[6]);
    runJob(input, output, measureClassName, t1, t2, convergenceDelta,
        maxIterations);
  }

  /**
   * Run the job
   *
   * @param input            the input pathname String
   * @param output           the output pathname String
   * @param measureClassName the DistanceMeasure class name
   * @param t1               the T1 distance threshold
   * @param t2               the T2 distance threshold
   * @param convergenceDelta the double convergence criteria
   * @param maxIterations    an int number of iterations
   */
  public static void runJob(String input, String output,
                            String measureClassName, double t1, double t2, double convergenceDelta,
                            int maxIterations) throws IOException {
    // delete the output directory
    JobConf conf = new JobConf(MeanShiftCanopyDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (fs.exists(outPath)) {
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);
    // iterate until the clusters converge
    boolean converged = false;
    int iteration = 0;
    String clustersIn = input;
    while (!converged && iteration < maxIterations) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String clustersOut = output + "/canopies-" + iteration;
      String controlOut = output + CONTROL_CONVERGED;
      MeanShiftCanopyDriver.runJob(clustersIn, clustersOut, controlOut,
          measureClassName, t1, t2, convergenceDelta);
      converged = FileSystem.get(conf).exists(new Path(controlOut));
      // now point the input to the old output directory
      clustersIn = output + "/canopies-" + iteration;
      iteration++;
    }
  }

}
