package org.apache.mahout.clustering.canopy;

/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


public class CanopyClusteringJob {

  /**
   * @param args
   */
  public static void main(String[] args) {
    String input = args[0];
    String output = args[1];
    String measureClassName = args[2];
    float t1 = new Float(args[3]);
    float t2 = new Float(args[4]);
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
                            String measureClassName, float t1, float t2) {
    CanopyDriver.runJob(input, output + "/canopies", measureClassName, t1, t2);
    ClusterDriver.runJob(input, output + "/canopies", output, measureClassName, t1, t2);
  }

}
