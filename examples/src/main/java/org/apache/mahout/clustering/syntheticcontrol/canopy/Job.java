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

import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends AbstractJob {

  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";

  private Job() {
  }

  private static final Logger log = LoggerFactory.getLogger(Job.class);

  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      ToolRunner.run(new Configuration(), new Job(), args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      HadoopUtil.delete(new Configuration(), output);
      run(new Path("testdata"), output, new EuclideanDistanceMeasure(), 80, 55);
    }
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
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param measure
   *          the DistanceMeasure to use
   * @param t1
   *          the canopy T1 threshold
   * @param t2
   *          the canopy T2 threshold
   */
  private static void run(Path input, Path output, DistanceMeasure measure,
      double t1, double t2) throws Exception {
    Path directoryContainingConvertedInput = new Path(output,
        DIRECTORY_CONTAINING_CONVERTED_INPUT);
    InputDriver.runJob(input, directoryContainingConvertedInput,
        "org.apache.mahout.math.RandomAccessSparseVector");
    CanopyDriver.run(new Configuration(), directoryContainingConvertedInput,
        output, measure, t1, t2, true, 0.0, false);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(new Path(output,
        "clusters-0-final"), new Path(output, "clusteredPoints"));
    clusterDumper.printClusters(null);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String, List<String>> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(new Configuration(), output);
    }
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
    double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);

    run(input, output, measure, t1, t2);
    return 0;
  }

}
