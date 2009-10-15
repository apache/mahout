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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class FuzzyKMeansJob {

  private static final Logger log = LoggerFactory
      .getLogger(FuzzyKMeansJob.class);

  private FuzzyKMeansJob() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption(obuilder, abuilder).create();
    Option outputOpt = DefaultOptionCreator.outputOption(obuilder, abuilder).create();
    Option convergenceDeltaOpt = DefaultOptionCreator.convergenceOption(obuilder, abuilder).create();
    Option measureClassOpt = DefaultOptionCreator.distanceOption(obuilder, abuilder).create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption(obuilder, abuilder).create();
    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Option clustersOpt = obuilder.withLongName("clusters").withRequired(true).withShortName("c").
        withArgument(abuilder.withName("clusters").withMinimum(1).withMaximum(1).create()).
        withDescription("The directory pathname for initial clusters.").create();
    
    Option numMapOpt = obuilder.withLongName("maxMap").withRequired(true).withShortName("p").
        withArgument(abuilder.withName("maxMap").withMinimum(1).withMaximum(1).create()).
        withDescription("The maximum number of maptasks.").create();
    
    Option numRedOpt = obuilder.withLongName("maxRed").withRequired(true).withShortName("r").
        withArgument(abuilder.withName("maxRed").withMinimum(1).withMaximum(1).create()).
        withDescription("The maximum number of reduce tasks.").create();
    
    Option doCanopyOpt = obuilder.withLongName("doCanopy").withRequired(true).withShortName("a").
        withArgument(abuilder.withName("doCanopy").withMinimum(1).withMaximum(1).create()).
        withDescription("Does canopy needed for initial clusters.").create();
    
    Option mOpt = obuilder.withLongName("fuzzify").withRequired(true).withShortName("m").
        withArgument(abuilder.withName("fuzzify").withMinimum(1).withMaximum(1).create()).
        withDescription("Param needed to fuzzify the cluster membership values.").create();

    Option vectorClassOpt = obuilder.withLongName("vectorclass").withRequired(true).withShortName("e").
        withArgument(abuilder.withName("vectorclass").withMinimum(1).withMaximum(1).create()).
        withDescription("Class name of vector implementation to use.").create();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(clustersOpt).
        withOption(outputOpt).withOption(measureClassOpt).withOption(convergenceDeltaOpt).
        withOption(maxIterOpt).withOption(numMapOpt).withOption(numRedOpt).withOption(doCanopyOpt).
        withOption(mOpt).withOption(vectorClassOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      String input = cmdLine.getValue(inputOpt).toString();
      String clusters = cmdLine.getValue(clustersOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();
      String measureClass = cmdLine.getValue(measureClassOpt).toString();
      double convergenceDelta = Double.parseDouble(cmdLine.getValue(convergenceDeltaOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      int numMapTasks = Integer.parseInt(cmdLine.getValue(numMapOpt).toString());
      int numReduceTasks = Integer.parseInt(cmdLine.getValue(numRedOpt).toString());
      boolean doCanopy = Boolean.parseBoolean(cmdLine.getValue(doCanopyOpt).toString());
      float m = Float.parseFloat(cmdLine.getValue(mOpt).toString());
      String vectorClassName = cmdLine.getValue(vectorClassOpt).toString();
      Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
      runJob(input, clusters, output, measureClass, convergenceDelta,
          maxIterations, numMapTasks, numReduceTasks, doCanopy, m, vectorClass);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
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
