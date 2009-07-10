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

package org.apache.mahout.clustering.fuzzykmeans;

import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.ManhattanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class FuzzyKMeansJob {

  private static final Logger log = LoggerFactory
      .getLogger(FuzzyKMeansJob.class);

  private FuzzyKMeansJob() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {

    if (args.length != 11) {
      log.warn("Expected num Arguments: 11  received: {}", args.length);
      printMessage();
      return;
    }
    int index = 0;
    String input = args[index++];
    String clusters = args[index++];
    String output = args[index++];
    String measureClass = args[index++];
    double convergenceDelta = Double.parseDouble(args[index++]);
    int maxIterations = Integer.parseInt(args[index++]);
    int numMapTasks = Integer.parseInt(args[index++]);
    int numReduceTasks = Integer.parseInt(args[index++]);
    boolean doCanopy = Boolean.parseBoolean(args[index++]);
    float m = Float.parseFloat(args[index++]);
    String vectorClassName = args[index++];
    Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
    runJob(input, clusters, output, measureClass, convergenceDelta,
        maxIterations, numMapTasks, numReduceTasks, doCanopy, m, vectorClass);
  }

  /** Prints Error Message */
  private static void printMessage() {
    log
        .warn("Usage: inputDir clusterDir OutputDir measureClass ConvergenceDelata  maxIterations numMapTasks numReduceTasks doCanopy m");
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input            the directory pathname for input points
   * @param clustersIn       the directory pathname for initial clusters
   * @param output           the directory pathname for output points
   * @param measureClass     the classname of the DistanceMeasure
   * @param convergenceDelta the convergence delta value
   * @param maxIterations    the maximum number of iterations
   * @param numMapTasks      the number of maptasks
   * @param doCanopy         does canopy needed for initial clusters
   * @param m                param needed to fuzzify the cluster membership values
   */
  public static void runJob(String input, String clustersIn, String output,
                            String measureClass, double convergenceDelta, int maxIterations,
                            int numMapTasks, int numReduceTasks, boolean doCanopy, float m, Class<? extends Vector> vectorClass)
      throws IOException {

    // run canopy to find initial clusters
    if (doCanopy) {
      CanopyDriver.runJob(input, clustersIn, ManhattanDistanceMeasure.class
          .getName(), 100.1, 50.1, vectorClass);

    }
    // run fuzzy k -means
    FuzzyKMeansDriver.runJob(input, clustersIn, output, measureClass,
        convergenceDelta, maxIterations, numMapTasks, numReduceTasks, m, vectorClass);

  }
}
