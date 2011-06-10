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

package org.apache.mahout.clustering.syntheticcontrol.meanshift;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.conversion.meanshift.InputDriver;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyDriver;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.kernel.IKernelProfile;
import org.apache.mahout.common.kernel.TriangularKernelProfile;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends AbstractJob {
  
  private static final Logger log = LoggerFactory.getLogger(Job.class);
  
  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";
  
  private Job() {}
  
  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      ToolRunner.run(new Configuration(), new Job(), args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      Configuration conf = new Configuration();
      HadoopUtil.delete(conf, output);
      new Job().run(conf, new Path("testdata"), output,
          new EuclideanDistanceMeasure(), new TriangularKernelProfile(), 47.6,
          1, 0.5, 10);
    }
  }
  
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException,
      InterruptedException, InstantiationException, IllegalAccessException {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.inputIsCanopiesOption().create());
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.kernelProfileOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    
    Map<String,String> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(new Configuration(), output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    String kernelProfileClass = getOption(DefaultOptionCreator.KERNEL_PROFILE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    double convergenceDelta = Double
        .parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer
        .parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    DistanceMeasure measure = ccl.loadClass(measureClass)
        .asSubclass(DistanceMeasure.class).newInstance();
    IKernelProfile kernelProfile = ccl.loadClass(kernelProfileClass)
        .asSubclass(IKernelProfile.class).newInstance();
    run(getConf(), input, output, measure, kernelProfile, t1, t2,
        convergenceDelta, maxIterations);
    return 0;
  }

  /**
   * Run the meanshift clustering job on an input dataset using the given
   * distance measure, t1, t2 and iteration parameters. All output data will be
   * written to the output directory, which will be initially deleted if it
   * exists. The clustered points will reside in the path
   * <output>/clustered-points. By default, the job expects the a file
   * containing synthetic_control.data as obtained from
   * http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
   * resides in a directory named "testdata", and writes output to a directory
   * named "output".
   * 
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param measure
   *          the DistanceMeasure to use
   * @param kernelProfile
   *          the IKernelProfile to use
   * @param t1
   *          the meanshift canopy T1 threshold
   * @param t2
   *          the meanshift canopy T2 threshold
   * @param convergenceDelta
   *          the double convergence criteria for iterations
   * @param maxIterations
   *          the int maximum number of iterations
   */
  public void run(Configuration conf, Path input, Path output,
      DistanceMeasure measure, IKernelProfile kernelProfile, double t1,
      double t2, double convergenceDelta, int maxIterations)
      throws IOException, InterruptedException, ClassNotFoundException {
    Path directoryContainingConvertedInput = new Path(output,
        DIRECTORY_CONTAINING_CONVERTED_INPUT);
    InputDriver.runJob(input, directoryContainingConvertedInput);
    MeanShiftCanopyDriver.run(conf, directoryContainingConvertedInput, output,
        measure, kernelProfile, t1, t2, convergenceDelta, maxIterations, true,
        true, false);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(new Path(output,
        "clusters-" + maxIterations), new Path(output, "clusteredPoints"));
    clusterDumper.printClusters(null);
  }
  
}
