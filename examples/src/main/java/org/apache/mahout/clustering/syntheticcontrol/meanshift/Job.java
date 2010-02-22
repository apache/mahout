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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopyJob;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job {

  private static final Logger log = LoggerFactory.getLogger(Job.class);
  
  private static final String CLUSTERED_POINTS_OUTPUT_DIRECTORY = "/clusteredPoints";
  
  private Job() {}
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption().withRequired(false).create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption().withRequired(false).create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption().withRequired(false).create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Option modelOpt = obuilder.withLongName("distanceClass").withRequired(false).withShortName("d")
        .withArgument(abuilder.withName("distanceClass").withMinimum(1).withMaximum(1).create())
        .withDescription("The distance measure class name.").create();
    
    Option threshold1Opt = obuilder.withLongName("threshold_1").withRequired(false).withShortName("t1")
        .withArgument(abuilder.withName("threshold_1").withMinimum(1).withMaximum(1).create())
        .withDescription("The T1 distance threshold.").create();
    
    Option threshold2Opt = obuilder.withLongName("threshold_2").withRequired(false).withShortName("t2")
        .withArgument(abuilder.withName("threshold_2").withMinimum(1).withMaximum(1).create())
        .withDescription("The T1 distance threshold.").create();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt)
        .withOption(modelOpt).withOption(helpOpt).withOption(convergenceDeltaOpt).withOption(threshold1Opt)
        .withOption(maxIterOpt).withOption(threshold2Opt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String input = cmdLine.getValue(inputOpt, "testdata").toString();
      String output = cmdLine.getValue(outputOpt, "output").toString();
      String measureClassName = cmdLine.getValue(modelOpt,
        "org.apache.mahout.common.distance.EuclideanDistanceMeasure").toString();
      double t1 = Double.parseDouble(cmdLine.getValue(threshold1Opt, "47.6").toString());
      double t2 = Double.parseDouble(cmdLine.getValue(threshold2Opt, "1").toString());
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt, "0.5").toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt, "10").toString());
      runJob(input, output, measureClassName, t1, t2, convergenceDelta, maxIterations);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  /**
   * Run the meanshift clustering job on an input dataset using the given distance measure, t1, t2 and
   * iteration parameters. All output data will be written to the output directory, which will be initially
   * deleted if it exists. The clustered points will reside in the path <output>/clustered-points. By default,
   * the job expects the a file containing synthetic_control.data as obtained from
   * http://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series resides in a directory named
   * "testdata", and writes output to a directory named "output".
   * 
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param measureClassName
   *          the String class name of the DistanceMeasure to use
   * @param t1
   *          the meanshift canopy T1 threshold
   * @param t2
   *          the meanshift canopy T2 threshold
   * @param convergenceDelta
   *          the double convergence criteria for iterations
   * @param maxIterations
   *          the int maximum number of iterations
   */
  private static void runJob(String input,
                             String output,
                             String measureClassName,
                             double t1,
                             double t2,
                             double convergenceDelta,
                             int maxIterations) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);
    
    Path outPath = new Path(output);
    client.setConf(conf);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    String directoryContainingConvertedInput = output + Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT;
    InputDriver.runJob(input, directoryContainingConvertedInput);
    MeanShiftCanopyJob.runJob(directoryContainingConvertedInput, output + "/meanshift", measureClassName, t1,
      t2, convergenceDelta, maxIterations);
    FileStatus[] status = dfs.listStatus(new Path(output + "/meanshift"));
    OutputDriver.runJob(status[status.length - 1].getPath().toString(), output
                                                                        + CLUSTERED_POINTS_OUTPUT_DIRECTORY);
  }
  
}
