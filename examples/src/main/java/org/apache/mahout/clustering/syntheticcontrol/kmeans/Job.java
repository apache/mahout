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

package org.apache.mahout.clustering.syntheticcontrol.kmeans;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.clustering.syntheticcontrol.canopy.InputDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job {

  private static final Logger log = LoggerFactory.getLogger(Job.class);

  private Job() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption().withRequired(false).create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption().withRequired(false).create();
    Option maxIterationsOpt = DefaultOptionCreator.maxIterOption().withRequired(false).create();

    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Distance Measure to use.  Default is SquaredEuclidean").withShortName("m").create();

    Option t1Opt = obuilder.withLongName("t1").withRequired(false).withArgument(
        abuilder.withName("t1").withMinimum(1).withMaximum(1).create()).withDescription("The t1 value to use.").withShortName("m")
        .create();
    Option t2Opt = obuilder.withLongName("t2").withRequired(false).withArgument(
        abuilder.withName("t2").withMinimum(1).withMaximum(1).create()).withDescription("The t2 value to use.").withShortName("m")
        .create();
    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Vector implementation class name.  Default is RandomAccessSparseVector.class").withShortName("v").create();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(measureClassOpt).withOption(
        convergenceDeltaOpt).withOption(maxIterationsOpt).withOption(vectorClassOpt).withOption(t1Opt).withOption(t2Opt)
        .withOption(helpOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      Path input = new Path(cmdLine.getValue(inputOpt, "testdata").toString());
      Path output = new Path(cmdLine.getValue(outputOpt, "output").toString());
      String measureClass = cmdLine.getValue(measureClassOpt, "org.apache.mahout.common.distance.EuclideanDistanceMeasure")
          .toString();
      double t1 = Double.parseDouble(cmdLine.getValue(t1Opt, "80").toString());
      double t2 = Double.parseDouble(cmdLine.getValue(t2Opt, "55").toString());
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt, "0.5").toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterationsOpt, 10).toString());
      // String className = cmdLine.getValue(vectorClassOpt,
      // "org.apache.mahout.math.RandomAccessSparseVector").toString();
      // Class<? extends Vector> vectorClass = Class.forName(className).asSubclass(Vector.class);

      runJob(input, output, measureClass, t1, t2, convergenceDelta, maxIterations);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the kmeans clustering job on an input dataset using the given distance measure, t1, t2 and iteration
   * parameters. All output data will be written to the output directory, which will be initially deleted if
   * it exists. The clustered points will reside in the path <output>/clustered-points. By default, the job
   * expects the a file containing synthetic_control.data as obtained from
   * http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series resides in a directory named
   * "testdata", and writes output to a directory named "output".
   * 
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param measureClass
   *          the String class name of the DistanceMeasure to use
   * @param t1
   *          the canopy T1 threshold
   * @param t2
   *          the canopy T2 threshold
   * @param convergenceDelta
   *          the double convergence criteria for iterations
   * @param maxIterations
   *          the int maximum number of iterations
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  private static void runJob(Path input, Path output, String measureClass, double t1, double t2, double convergenceDelta,
      int maxIterations) throws IOException, InstantiationException, IllegalAccessException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);

    client.setConf(conf);
    HadoopUtil.overwriteOutput(output);

    Path directoryContainingConvertedInput = new Path(output, Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT);
    log.info("Preparing Input");
    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
    log.info("Running Canopy to get initial clusters");
    CanopyDriver.runJob(directoryContainingConvertedInput, output, measureClass, t1, t2, false);
    log.info("Running KMeans");
    KMeansDriver.runJob(directoryContainingConvertedInput, new Path(output, Cluster.INITIAL_CLUSTERS_DIR), output, measureClass,
        convergenceDelta, maxIterations, 1, true);

    ClusterDumper clusterDumper =
        new ClusterDumper(new Path(output, "clusters-10"), new Path(output, "clusteredPoints"));
    clusterDumper.printClusters(null);
  }
}
