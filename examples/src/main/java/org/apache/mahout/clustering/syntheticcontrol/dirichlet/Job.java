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

package org.apache.mahout.clustering.syntheticcontrol.dirichlet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
import org.apache.hadoop.mapred.JobConf;
import org.apache.log4j.Logger;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.DirichletJob;
import org.apache.mahout.clustering.dirichlet.DirichletMapper;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.syntheticcontrol.canopy.InputDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.SparseVector;

import static org.apache.mahout.clustering.syntheticcontrol.Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT;

public class Job {

  /**Logger for this class.*/
  private static final Logger LOG = Logger.getLogger(Job.class);

  private Job() {
  }

  public static void main(String[] args) throws IOException,
      ClassNotFoundException, InstantiationException, IllegalAccessException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption(obuilder, abuilder).withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption(obuilder, abuilder).withRequired(false).create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption(obuilder, abuilder).withRequired(false).create();
    Option topicsOpt = DefaultOptionCreator.kOption(obuilder, abuilder).withRequired(false).create();

    Option redOpt = obuilder.withLongName("reducerNum").withRequired(false).withArgument(
        abuilder.withName("r").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of reducers to use.").withShortName("r").create();

    Option vectorOpt = obuilder.withLongName("vector").withRequired(false).withArgument(
        abuilder.withName("v").withMinimum(1).withMaximum(1).create()).withDescription(
        "The vector implementation to use.").withShortName("v").create();

    Option mOpt = obuilder.withLongName("alpha").withRequired(false).withShortName("m").
        withArgument(abuilder.withName("alpha").withMinimum(1).withMaximum(1).create()).
        withDescription("The alpha0 value for the DirichletDistribution.").create();

    Option modelOpt = obuilder.withLongName("modelClass").withRequired(false).withShortName("d").
        withArgument(abuilder.withName("modelClass").withMinimum(1).withMaximum(1).create()).
          withDescription("The ModelDistribution class name.").create();
    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(modelOpt).
        withOption(maxIterOpt).withOption(mOpt).withOption(topicsOpt).withOption(redOpt).withOption(helpOpt).create();

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
      String modelFactory = cmdLine.getValue(modelOpt, "org.apache.mahout.clustering.syntheticcontrol.dirichlet.NormalScModelDistribution").toString();
      int numModels = Integer.parseInt(cmdLine.getValue(topicsOpt, "10").toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt, "5").toString());
      double alpha_0 = Double.parseDouble(cmdLine.getValue(mOpt, "1.0").toString());
      int numReducers = Integer.parseInt(cmdLine.getValue(redOpt, "1").toString());
      String vectorClassName = cmdLine.getValue(vectorOpt, "org.apache.mahout.matrix.SparseVector").toString();
      Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
      runJob(input, output, modelFactory, numModels, maxIterations, alpha_0, numReducers, vectorClass);
    } catch (OptionException e) {
      LOG.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments, deleting the output directory if it
   * exists beforehand
   * 
   * @param input the directory pathname for input points
   * @param output the directory pathname for output points
   * @param modelFactory the ModelDistribution class name
   * @param numModels the number of Models
   * @param maxIterations the maximum number of iterations
   * @param alpha_0 the alpha0 value for the DirichletDistribution
   * @param numReducers the desired number of reducers
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   * @throws ClassNotFoundException 
   */
  public static void runJob(String input, String output, String modelFactory,
      int numModels, int maxIterations, double alpha_0, int numReducers, Class<? extends Vector> vectorClass)
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
    final String directoryContainingConvertedInput = output + DIRECTORY_CONTAINING_CONVERTED_INPUT;
    InputDriver.runJob(input, directoryContainingConvertedInput, vectorClass);
    DirichletDriver.runJob(directoryContainingConvertedInput, output + "/state", modelFactory,
        numModels, maxIterations, alpha_0, numReducers);
    printResults(output + "/state", modelFactory, maxIterations, numModels,
        alpha_0);
  }

  /**
   * Prints out all of the clusters during each iteration
   * @param output the String output directory
   * @param modelDistribution the String class name of the ModelDistribution
   * @param numIterations the int number of Iterations
   * @param numModels the int number of models
   * @param alpha_0 the double alpha_0 value
   */
  public static void printResults(String output, String modelDistribution,
      int numIterations, int numModels, double alpha_0) {
    List<List<DirichletCluster<Vector>>> clusters = new ArrayList<List<DirichletCluster<Vector>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY, modelDistribution);
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(numModels));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(alpha_0));
    for (int i = 0; i < numIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, output + "/state-" + i);
      clusters.add(DirichletMapper.getDirichletState(conf).clusters);
    }
    printResults(clusters, 0);

  }

  /**
   * Actually prints out the clusters
   * @param clusters a List of Lists of DirichletClusters
   * @param significant the minimum number of samples to enable printing a model
   */
  private static void printResults(
      List<List<DirichletCluster<Vector>>> clusters, int significant) {
    int row = 0;
    for (List<DirichletCluster<Vector>> r : clusters) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.size(); k++) {
        Model<Vector> model = r.get(k).model;
        if (model.count() > significant) {
          int total = (int) r.get(k).totalCount;
          System.out.print("m" + k + '(' + total + ')' + model.toString()
              + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }
}
