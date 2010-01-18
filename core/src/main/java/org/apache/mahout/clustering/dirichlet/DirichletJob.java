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

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirichletJob {

  private static final Logger log = LoggerFactory.getLogger(DirichletJob.class);

  private DirichletJob() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException, InstantiationException,
      IllegalAccessException, SecurityException, IllegalArgumentException, NoSuchMethodException, InvocationTargetException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption().create();
    Option topicsOpt = DefaultOptionCreator.kOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();

    Option mOpt = obuilder.withLongName("alpha").withRequired(true).withShortName("m").withArgument(
        abuilder.withName("alpha").withMinimum(1).withMaximum(1).create()).withDescription(
        "The alpha0 value for the DirichletDistribution.").create();

    Option modelOpt = obuilder.withLongName("modelClass").withRequired(true).withShortName("d").withArgument(
        abuilder.withName("modelClass").withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution class name.").create();

    Option prototypeOpt = obuilder.withLongName("modelPrototypeClass").withRequired(true).withShortName("p").withArgument(
        abuilder.withName("prototypeClass").withMinimum(1).withMaximum(1).create()).withDescription(
        "The ModelDistribution prototype Vector class name.").create();

    Option sizeOpt = obuilder.withLongName("prototypeSize").withRequired(true).withShortName("s").withArgument(
        abuilder.withName("prototypeSize").withMinimum(1).withMaximum(1).create()).withDescription(
        "The ModelDistribution prototype Vector size.").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(modelOpt).withOption(
        prototypeOpt).withOption(sizeOpt).withOption(maxIterOpt).withOption(mOpt).withOption(topicsOpt).withOption(helpOpt)
        .create();

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
      String modelFactory = cmdLine.getValue(modelOpt).toString();
      String modelPrototype = cmdLine.getValue(prototypeOpt).toString();
      int prototypeSize = Integer.parseInt(cmdLine.getValue(sizeOpt).toString());
      int numModels = Integer.parseInt(cmdLine.getValue(topicsOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      double alpha_0 = Double.parseDouble(cmdLine.getValue(mOpt).toString());
      runJob(input, output, modelFactory, modelPrototype, prototypeSize, numModels, maxIterations, alpha_0);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }

  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it exists beforehand
   *
   * @param input         the directory pathname for input points
   * @param output        the directory pathname for output points
   * @param modelFactory  the ModelDistribution class name
   * @param modelPrototype the Vector class name used by the modelFactory
   * @param prototypeSize the size of the prototype vector
   * @param numModels     the number of Models
   * @param maxIterations the maximum number of iterations
   * @param alpha_0       the alpha0 value for the DirichletDistribution
   * @throws InvocationTargetException 
   * @throws NoSuchMethodException 
   * @throws IllegalArgumentException 
   * @throws SecurityException 
   */
  public static void runJob(String input, String output, String modelFactory, String modelPrototype, int prototypeSize,
      int numModels, int maxIterations, double alpha_0) throws IOException, ClassNotFoundException, InstantiationException,
      IllegalAccessException, SecurityException, IllegalArgumentException, NoSuchMethodException, InvocationTargetException {
    // delete the output directory
    Configuration conf = new JobConf(DirichletJob.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (fs.exists(outPath)) {
      fs.delete(outPath, true);
    }
    fs.mkdirs(outPath);
    DirichletDriver.runJob(input, output, modelFactory, modelPrototype, prototypeSize, numModels, maxIterations, alpha_0, 1);
  }
}
