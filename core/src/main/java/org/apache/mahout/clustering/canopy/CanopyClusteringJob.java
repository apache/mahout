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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.CommandLineUtil;
import org.apache.mahout.utils.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Runs the {@link org.apache.mahout.clustering.canopy.CanopyDriver#runJob(String, String, String, double, double,
 * Class)} and then {@link org.apache.mahout.clustering.canopy.ClusterDriver#runJob(String, String, String, String,
 * double, double, Class)}.
 */
public final class CanopyClusteringJob {

  private static final Logger log = LoggerFactory.getLogger(CanopyClusteringJob.class);

  /** The default name of the canopies output sub-directory. */
  public static final String DEFAULT_CANOPIES_OUTPUT_DIRECTORY = "/canopies";
  /** The default name of the directory used to output clusters. */
  public static final String DEFAULT_CLUSTER_OUTPUT_DIRECTORY = ClusterDriver.DEFAULT_CLUSTER_OUTPUT_DIRECTORY;

  private CanopyClusteringJob() {
  }

  /**
   * @param args
   */
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
        withDescription("The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
        withDescription("The Path to put the output in").withShortName("o").create();

    Option measureClassOpt = obuilder.withLongName("distance").withRequired(false).withArgument(
        abuilder.withName("distance").withMinimum(1).withMaximum(1).create()).
        withDescription("The Distance Measure to use.  Default is SquaredEuclidean").withShortName("m").create();

    Option vectorClassOpt = obuilder.withLongName("vectorClass").withRequired(false).withArgument(
        abuilder.withName("vectorClass").withMinimum(1).withMaximum(1).create()).
        withDescription("The Vector implementation class name.  Default is SparseVector.class").withShortName("v").create();
    Option t1Opt = obuilder.withLongName("t1").withRequired(true).withArgument(
        abuilder.withName("t1").withMinimum(1).withMaximum(1).create()).
        withDescription("t1").withShortName("t1").create();
    Option t2Opt = obuilder.withLongName("t2").withRequired(true).withArgument(
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

      String input = cmdLine.getValue(inputOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      String measureClass = SquaredEuclideanDistanceMeasure.class.getName();
      if (cmdLine.hasOption(measureClassOpt)) {
        measureClass = cmdLine.getValue(measureClassOpt).toString();
      }

      Class<? extends Vector> vectorClass = cmdLine.hasOption(vectorClassOpt) == false ?
          SparseVector.class
          : (Class<? extends Vector>) Class.forName(cmdLine.getValue(vectorClassOpt).toString());
      double t1 = Double.parseDouble(cmdLine.getValue(t1Opt).toString());
      double t2 = Double.parseDouble(cmdLine.getValue(t2Opt).toString());

      runJob(input, output, measureClass, t1, t2, vectorClass);

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
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
                            String measureClassName, double t1, double t2, Class<? extends Vector> vectorClass) throws IOException {
    CanopyDriver.runJob(input, output + DEFAULT_CANOPIES_OUTPUT_DIRECTORY, measureClassName, t1, t2, vectorClass);
    ClusterDriver.runJob(input, output + DEFAULT_CANOPIES_OUTPUT_DIRECTORY, output, measureClassName, t1, t2, vectorClass);
  }

}
