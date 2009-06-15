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

import java.io.IOException;

public class CanopyClusteringJob {

  /**
   * The default name of the canopies output sub-directory.
   */     
  public static final String DEFAULT_CANOPIES_OUTPUT_DIRECTORY = "/canopies";
  /**
   * The default name of the directory used to output clusters.
   */
  public static final String DEFAULT_CLUSTER_OUTPUT_DIRECTORY = ClusterDriver.DEFAULT_CLUSTER_OUTPUT_DIRECTORY;

  private CanopyClusteringJob() {
  }

  /**
   * @param args
   */
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
    CanopyDriver.runJob(input, output + DEFAULT_CANOPIES_OUTPUT_DIRECTORY, measureClassName, t1, t2);
    ClusterDriver.runJob(input, output + DEFAULT_CANOPIES_OUTPUT_DIRECTORY, output, measureClassName, t1, t2);
  }

}
