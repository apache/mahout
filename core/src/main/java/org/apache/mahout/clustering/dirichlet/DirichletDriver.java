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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class DirichletDriver {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String MODEL_FACTORY_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";

  private static final Logger log = LoggerFactory
      .getLogger(DirichletDriver.class);

  private DirichletDriver() {
  }

  public static void main(String[] args) throws InstantiationException,
      IllegalAccessException, ClassNotFoundException, IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption(obuilder, abuilder).create();
    Option outputOpt = DefaultOptionCreator.outputOption(obuilder, abuilder).create();
    Option maxIterOpt = DefaultOptionCreator.maxIterOption(obuilder, abuilder).create();
    Option topicsOpt = DefaultOptionCreator.kOption(obuilder, abuilder).create();
    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Option mOpt = obuilder.withLongName("alpha").withRequired(true).withShortName("m").
        withArgument(abuilder.withName("alpha").withMinimum(1).withMaximum(1).create()).
        withDescription("The alpha0 value for the DirichletDistribution.").create();

    Option modelOpt = obuilder.withLongName("modelClass").withRequired(true).withShortName("d").
        withArgument(abuilder.withName("modelClass").withMinimum(1).withMaximum(1).create()).
        withDescription("The ModelDistribution class name.").create();

    Option numRedOpt = obuilder.withLongName("maxRed").withRequired(true).withShortName("r").
        withArgument(abuilder.withName("maxRed").withMinimum(1).withMaximum(1).create()).
        withDescription("The number of reduce tasks.").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(modelOpt).
        withOption(maxIterOpt).withOption(mOpt).withOption(topicsOpt).withOption(helpOpt).
        withOption(numRedOpt).create();

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
      int numReducers = Integer.parseInt(cmdLine.getValue(numRedOpt).toString());
      int numModels = Integer.parseInt(cmdLine.getValue(topicsOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      double alpha_0 = Double.parseDouble(cmdLine.getValue(mOpt).toString());
      runJob(input, output, modelFactory, numModels, maxIterations, alpha_0, numReducers);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input         the directory pathname for input points
   * @param output        the directory pathname for output points
   * @param modelFactory  the String ModelDistribution class name to use
   * @param numClusters   the number of models
   * @param maxIterations the maximum number of iterations
   * @param alpha_0       the alpha_0 value for the DirichletDistribution
   * @param numReducers   the number of Reducers desired
   */
  public static void runJob(String input, String output, String modelFactory,
                            int numClusters, int maxIterations, double alpha_0, int numReducers)
      throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException {

    String stateIn = output + "/state-0";
    writeInitialState(output, stateIn, modelFactory, numClusters, alpha_0);

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String stateOut = output + "/state-" + (iteration + 1);
      runIteration(input, stateIn, stateOut, modelFactory, numClusters,
          alpha_0, numReducers);
      // now point the input to the old output directory
      stateIn = stateOut;
    }
  }

  private static void writeInitialState(String output, String stateIn,
                                        String modelFactory, int numModels, double alpha_0)
      throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException {
    DirichletState<Vector> state = createState(modelFactory, numModels, alpha_0);
    JobConf job = new JobConf(KMeansDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), job);
    fs.delete(outPath, true);
    for (int i = 0; i < numModels; i++) {
      Path path = new Path(stateIn + "/part-" + i);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path,
          Text.class, DirichletCluster.class);
      writer.append(new Text(Integer.toString(i)), state.getClusters().get(i));
      writer.close();
    }
  }

  @SuppressWarnings("unchecked")
  public static DirichletState<Vector> createState(String modelFactory,
                                                   int numModels, double alpha_0) throws ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(modelFactory);
    ModelDistribution<Vector> factory = (ModelDistribution<Vector>) cl
        .newInstance();
    return new DirichletState<Vector>(factory,
        numModels, alpha_0, 1, 1);
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input        the directory pathname for input points
   * @param stateIn      the directory pathname for input state
   * @param stateOut     the directory pathname for output state
   * @param modelFactory the class name of the model factory class
   * @param numClusters  the number of clusters
   * @param alpha_0      alpha_0
   * @param numReducers  the number of Reducers desired
   */
  public static void runIteration(String input, String stateIn,
                                  String stateOut, String modelFactory, int numClusters, double alpha_0,
                                  int numReducers) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(DirichletDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DirichletCluster.class);
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(SparseVector.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(stateOut);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(DirichletMapper.class);
    conf.setReducerClass(DirichletReducer.class);
    conf.setNumReduceTasks(numReducers);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(STATE_IN_KEY, stateIn);
    conf.set(MODEL_FACTORY_KEY, modelFactory);
    conf.set(NUM_CLUSTERS_KEY, Integer.toString(numClusters));
    conf.set(ALPHA_0_KEY, Double.toString(alpha_0));

    client.setConf(conf);
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input   the directory pathname for input points
   * @param stateIn the directory pathname for input state
   * @param output  the directory pathname for output points
   */
  public static void runClustering(String input, String stateIn, String output) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(DirichletDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);

    conf.setMapperClass(DirichletMapper.class);
    conf.setNumReduceTasks(0);

    client.setConf(conf);
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }
}
