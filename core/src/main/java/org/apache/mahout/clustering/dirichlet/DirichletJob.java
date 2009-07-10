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

package org.apache.mahout.clustering.dirichlet;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import java.io.IOException;

public class DirichletJob {

  private DirichletJob() {
  }

  public static void main(String[] args) throws IOException,
      ClassNotFoundException, InstantiationException, IllegalAccessException {
    String input = args[0];
    String output = args[1];
    String modelFactory = args[2];
    int numModels = Integer.parseInt(args[3]);
    int maxIterations = Integer.parseInt(args[4]);
    double alpha_0 = Double.parseDouble(args[5]);
    runJob(input, output, modelFactory, numModels, maxIterations, alpha_0);
  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it exists beforehand
   *
   * @param input         the directory pathname for input points
   * @param output        the directory pathname for output points
   * @param modelFactory  the ModelDistribution class name
   * @param numModels     the number of Models
   * @param maxIterations the maximum number of iterations
   * @param alpha_0       the alpha0 value for the DirichletDistribution
   */
  public static void runJob(String input, String output, String modelFactory,
                            int numModels, int maxIterations, double alpha_0)
      throws IOException, ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    // delete the output directory
    JobConf conf = new JobConf(DirichletJob.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (fs.exists(outPath)) {
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);
    DirichletDriver.runJob(input, output, modelFactory, numModels, maxIterations,
        alpha_0, 1);
  }
}
