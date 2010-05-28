/*
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
import java.lang.reflect.InvocationTargetException;
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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.dirichlet.DirichletCluster;
import org.apache.mahout.clustering.dirichlet.DirichletDriver;
import org.apache.mahout.clustering.dirichlet.DirichletMapper;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.syntheticcontrol.Constants;
import org.apache.mahout.clustering.syntheticcontrol.canopy.InputDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job {
  
  private static final Logger log = LoggerFactory.getLogger(Job.class);
  
  private Job() { }
  
  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption().withRequired(false).create();
    Option maxIterOpt = DefaultOptionCreator.maxIterationsOption().withRequired(false).create();
    Option topicsOpt = DefaultOptionCreator.kOption().withRequired(false).create();
    
    Option redOpt = obuilder.withLongName("reducerNum").withRequired(false).withArgument(
      abuilder.withName("r").withMinimum(1).withMaximum(1).create()).withDescription(
      "The number of reducers to use.").withShortName("r").create();
    
    Option vectorOpt = obuilder.withLongName("vector").withRequired(false).withArgument(
      abuilder.withName("v").withMinimum(1).withMaximum(1).create()).withDescription(
      "The vector implementation to use.").withShortName("v").create();
    
    Option mOpt = obuilder.withLongName("alpha").withRequired(false).withShortName("m").withArgument(
      abuilder.withName("alpha").withMinimum(1).withMaximum(1).create()).withDescription(
      "The alpha0 value for the DirichletDistribution.").create();
    
    Option modelOpt = obuilder.withLongName("modelClass").withRequired(false).withShortName("d")
        .withArgument(abuilder.withName("modelClass").withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution class name.").create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt)
        .withOption(modelOpt).withOption(maxIterOpt).withOption(mOpt).withOption(topicsOpt)
        .withOption(redOpt).withOption(helpOpt).create();
    
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
      String modelFactory = cmdLine.getValue(modelOpt,
        "org.apache.mahout.clustering.syntheticcontrol.dirichlet.NormalScModelDistribution").toString();
      int numModels = Integer.parseInt(cmdLine.getValue(topicsOpt, "10").toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt, "5").toString());
      double alpha0 = Double.parseDouble(cmdLine.getValue(mOpt, "1.0").toString());
      int numReducers = Integer.parseInt(cmdLine.getValue(redOpt, "1").toString());
      String vectorClassName = cmdLine.getValue(vectorOpt, "org.apache.mahout.math.RandomAccessSparseVector")
          .toString();
      runJob(input, output, modelFactory, numModels, maxIterations, alpha0, numReducers,
            vectorClassName);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  /**
   * Run the job using supplied arguments, deleting the output directory if it exists beforehand
   * 
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   * @param modelFactory
   *          the ModelDistribution class name
   * @param numModels
   *          the number of Models
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha0 value for the DirichletDistribution
   * @param numReducers
   *          the desired number of reducers
   */
  public static void runJob(Path input,
                            Path output,
                            String modelFactory,
                            int numModels,
                            int maxIterations,
                            double alpha0,
                            int numReducers,
                            String vectorClassName) throws IOException,
                                                   ClassNotFoundException,
                                                   InstantiationException,
                                                   IllegalAccessException,
                                                   SecurityException,
                                                   IllegalArgumentException,
                                                   NoSuchMethodException,
                                                   InvocationTargetException {
    HadoopUtil.overwriteOutput(output);

    Path directoryContainingConvertedInput = new Path(output, Constants.DIRECTORY_CONTAINING_CONVERTED_INPUT);
    InputDriver.runJob(input, directoryContainingConvertedInput, vectorClassName);
    DirichletDriver.runJob(directoryContainingConvertedInput, output, modelFactory,
      vectorClassName, numModels, maxIterations, alpha0, numReducers, true, true, 0);
  }
  
  /**
   * Prints out all of the clusters during each iteration
   * 
   * @param output
   *          the String output directory
   * @param modelDistribution
   *          the String class name of the ModelDistribution
   * @param vectorClassName
   *          the String class name of the Vector to use
   * @param prototypeSize
   *          the size of the Vector prototype for the Dirichlet Models
   * @param numIterations
   *          the int number of Iterations
   * @param numModels
   *          the int number of models
   * @param alpha0
   *          the double alpha_0 value
   * @throws InvocationTargetException
   * @throws NoSuchMethodException
   * @throws SecurityException
   */
  public static void printResults(String output,
                                  String modelDistribution,
                                  String vectorClassName,
                                  int prototypeSize,
                                  int numIterations,
                                  int numModels,
                                  double alpha0) throws SecurityException,
                                                 NoSuchMethodException,
                                                 InvocationTargetException {
    List<List<DirichletCluster<VectorWritable>>> clusters = new ArrayList<List<DirichletCluster<VectorWritable>>>();
    JobConf conf = new JobConf(KMeansDriver.class);
    conf.set(DirichletDriver.MODEL_FACTORY_KEY, modelDistribution);
    conf.set(DirichletDriver.NUM_CLUSTERS_KEY, Integer.toString(numModels));
    conf.set(DirichletDriver.ALPHA_0_KEY, Double.toString(alpha0));
    for (int i = 0; i < numIterations; i++) {
      conf.set(DirichletDriver.STATE_IN_KEY, output + "/state-" + i);
      conf.set(DirichletDriver.MODEL_PROTOTYPE_KEY, vectorClassName);
      conf.set(DirichletDriver.PROTOTYPE_SIZE_KEY, Integer.toString(prototypeSize));
      clusters.add(DirichletMapper.getDirichletState(conf).getClusters());
    }
    printResults(clusters, 0);
    
  }
  
  /**
   * Actually prints out the clusters
   * 
   * @param clusters
   *          a List of Lists of DirichletClusters
   * @param significant
   *          the minimum number of samples to enable printing a model
   */
  private static void printResults(List<List<DirichletCluster<VectorWritable>>> clusters, int significant) {
    int row = 0;
    StringBuilder result = new StringBuilder();
    for (List<DirichletCluster<VectorWritable>> r : clusters) {
      result.append("sample=").append(row++).append("]= ");
      for (int k = 0; k < r.size(); k++) {
        Model<VectorWritable> model = r.get(k).getModel();
        if (model.count() > significant) {
          int total = (int) r.get(k).getTotalCount();
          result.append('m').append(k).append('(').append(total).append(')').append(model).append(", ");
        }
      }
      result.append('\n');
    }
    result.append('\n');
    log.info(result.toString());
  }
}
