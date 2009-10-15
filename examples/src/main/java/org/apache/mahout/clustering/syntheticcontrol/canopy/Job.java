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

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.log4j.Logger;
import org.apache.mahout.clustering.canopy.CanopyClusteringJob;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.matrix.Vector;

public class Job {
  /** Logger for this class.*/
  private static final Logger LOG = Logger.getLogger(Job.class);

  private Job() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
      DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
      ArgumentBuilder abuilder = new ArgumentBuilder();
      GroupBuilder gbuilder = new GroupBuilder();

      Option inputOpt = obuilder.withLongName("input").withRequired(false).withArgument(
          abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
          withDescription("The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();
      Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
          withDescription("The Path to put the output in").withShortName("o").create();

      Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
          abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).
          withDescription("The Distance Measure to use.  Default is SquaredEuclidean").withShortName("m").create();
      Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
          abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).
          withDescription("The Vector implementation class name.  Default is SparseVector.class").withShortName("v").create();

      Option t1Opt = obuilder.withLongName("t1").withRequired(false).withArgument(
          abuilder.withName("t1").withMinimum(1).withMaximum(1).create()).
          withDescription("t1").withShortName("t1").create();
      Option t2Opt = obuilder.withLongName("t2").withRequired(false).withArgument(
          abuilder.withName("t2").withMinimum(1).withMaximum(1).create()).
          withDescription("t2").withShortName("t2").create();


      Option helpOpt = obuilder.withLongName("help").
          withDescription("Print out help").withShortName("h").create();

      Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt)
          .withOption(measureClassOpt).withOption(vectorClassOpt)
          .withOption(t1Opt).withOption(t2Opt)
          .withOption(helpOpt).create();


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
        String measureClass = cmdLine.getValue(
            measureClassOpt, "org.apache.mahout.common.distance.EuclideanDistanceMeasure").toString();

        Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(
            cmdLine.getValue(vectorClassOpt, "org.apache.mahout.matrix.SparseVector").toString());
        double t1 = Double.parseDouble(cmdLine.getValue(t1Opt, "80").toString());
        double t2 = Double.parseDouble(cmdLine.getValue(t2Opt, "55").toString());

        runJob(input, output, measureClass, t1, t2, vectorClass);
      } catch (OptionException e) {
        LOG.error("Exception", e);
        CommandLineUtil.printHelp(group);
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
   * @param measureClassName
   *          the String class name of the DistanceMeasure to use
   * @param t1
   *          the canopy T1 threshold
   * @param t2
   *          the canopy T2 threshold
   */
  private static void runJob(String input, String output,
      String measureClassName, double t1, double t2,
      Class<? extends Vector> vectorClass) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);

    Path outPath = new Path(output);
    client.setConf(conf);
    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath))
      dfs.delete(outPath, true);
    final String directoryContainingConvertedInput = output
        + Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT;
    InputDriver.runJob(input, directoryContainingConvertedInput, vectorClass);
    CanopyClusteringJob.runJob(directoryContainingConvertedInput, output,
        measureClassName, t1, t2, vectorClass);
  }

}
