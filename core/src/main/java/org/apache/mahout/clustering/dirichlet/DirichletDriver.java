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
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configurable;
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
import org.apache.mahout.clustering.dirichlet.models.VectorModelDistribution;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirichletDriver {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String MODEL_FACTORY_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String MODEL_PROTOTYPE_KEY = "org.apache.mahout.clustering.dirichlet.modelPrototype";

  public static final String PROTOTYPE_SIZE_KEY = "org.apache.mahout.clustering.dirichlet.prototypeSize";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";

  private static final Logger log = LoggerFactory.getLogger(DirichletDriver.class);

  private DirichletDriver() {
  }

  public static void main(String[] args) throws InstantiationException, IllegalAccessException, ClassNotFoundException,
      IOException, SecurityException, IllegalArgumentException, NoSuchMethodException, InvocationTargetException {
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

    Option numRedOpt = obuilder.withLongName("maxRed").withRequired(true).withShortName("r").withArgument(
        abuilder.withName("maxRed").withMinimum(1).withMaximum(1).create()).withDescription("The number of reduce tasks.").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(modelOpt).withOption(
        maxIterOpt).withOption(mOpt).withOption(topicsOpt).withOption(helpOpt).withOption(numRedOpt).create();

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
   * @throws InvocationTargetException 
   * @throws NoSuchMethodException 
   * @throws IllegalArgumentException 
   * @throws SecurityException 
   * @deprecated since it presumes 2-d, dense vector model prototypes
   */
  public static void runJob(String input, String output, String modelFactory, int numClusters, int maxIterations, double alpha_0,
      int numReducers) throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException,
      SecurityException, IllegalArgumentException, NoSuchMethodException, InvocationTargetException {
    runJob(input, output, modelFactory, "org.apache.mahout.math.DenseVector", 2, numClusters, maxIterations, alpha_0, numReducers);
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
   * @throws InvocationTargetException 
   * @throws NoSuchMethodException 
   * @throws IllegalArgumentException 
   * @throws SecurityException 
   */
  public static void runJob(String input, String output, String modelFactory, String modelPrototype, int prototypeSize,
      int numClusters, int maxIterations, double alpha_0, int numReducers) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException, SecurityException, IllegalArgumentException, NoSuchMethodException,
      InvocationTargetException {

    String stateIn = output + "/state-0";
    writeInitialState(output, stateIn, modelFactory, modelPrototype, prototypeSize, numClusters, alpha_0);

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String stateOut = output + "/state-" + (iteration + 1);
      runIteration(input, stateIn, stateOut, modelFactory, modelPrototype, prototypeSize, numClusters, alpha_0, numReducers);
      // now point the input to the old output directory
      stateIn = stateOut;
    }
  }

  private static void writeInitialState(String output, String stateIn, String modelFactory, String modelPrototype,
      int prototypeSize, int numModels, double alpha_0) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, IOException, SecurityException, IllegalArgumentException, NoSuchMethodException,
      InvocationTargetException {

    DirichletState<VectorWritable> state = createState(modelFactory, modelPrototype, prototypeSize, numModels, alpha_0);
    JobConf job = new JobConf(KMeansDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), job);
    fs.delete(outPath, true);
    for (int i = 0; i < numModels; i++) {
      Path path = new Path(stateIn + "/part-" + i);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, Text.class, DirichletCluster.class);
      writer.append(new Text(Integer.toString(i)), state.getClusters().get(i));
      writer.close();
    }
  }

  /**
   * Creates a DirichletState object from the given arguments. Note that the modelFactory
   * is presumed to be a subclass of VectorModelDistribution that can be initialized with
   * a concrete Vector prototype.
   * 
   * @param modelFactory a String which is the class name of the model factory
   * @param modelPrototype a String which is the class name of the Vector used to initialize the factory
   * @param prototypeSie an int number of dimensions of the model prototype vector
   * @param numModels an int number of models to be created
   * @param alpha_0 the double alpha_0 argument to the algorithm
   * @return an initialized DirichletState
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws NoSuchMethodException 
   * @throws SecurityException 
   * @throws InvocationTargetException 
   * @throws IllegalArgumentException 
   */
  public static DirichletState<VectorWritable> createState(String modelFactory, String modelPrototype, int prototypeSize,
      int numModels, double alpha_0) throws ClassNotFoundException, InstantiationException, IllegalAccessException,
      SecurityException, NoSuchMethodException, IllegalArgumentException, InvocationTargetException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<? extends VectorModelDistribution> cl = ccl.loadClass(modelFactory).asSubclass(VectorModelDistribution.class);
    VectorModelDistribution factory = (VectorModelDistribution) cl.newInstance();

    Class<? extends Vector> vcl = ccl.loadClass(modelPrototype).asSubclass(Vector.class);
    Constructor<? extends Vector> v = vcl.getConstructor(int.class);
    factory.setModelPrototype(new VectorWritable(v.newInstance(prototypeSize)));
    return new DirichletState<VectorWritable>(factory, numModels, alpha_0, 1, 1);
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input        the directory pathname for input points
   * @param stateIn      the directory pathname for input state
   * @param stateOut     the directory pathname for output state
   * @param modelFactory the class name of the model factory class
   * @param modelPrototype TODO
   * @param prototypeSize TODO
   * @param numClusters  the number of clusters
   * @param alpha_0      alpha_0
   * @param numReducers  the number of Reducers desired
   */
  public static void runIteration(String input, String stateIn, String stateOut, String modelFactory, String modelPrototype,
      int prototypeSize, int numClusters, double alpha_0, int numReducers) {
    Configurable client = new JobClient();
    JobConf conf = new JobConf(DirichletDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DirichletCluster.class);
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(VectorWritable.class);

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
    conf.set(MODEL_PROTOTYPE_KEY, modelPrototype);
    conf.set(PROTOTYPE_SIZE_KEY, Integer.toString(prototypeSize));
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
    Configurable client = new JobClient();
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
