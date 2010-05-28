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
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputLogFilter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.dirichlet.models.VectorModelDistribution;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class DirichletDriver {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String MODEL_FACTORY_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String MODEL_PROTOTYPE_KEY = "org.apache.mahout.clustering.dirichlet.modelPrototype";

  public static final String PROTOTYPE_SIZE_KEY = "org.apache.mahout.clustering.dirichlet.prototypeSize";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";

  public static final String EMIT_MOST_LIKELY_KEY = "org.apache.mahout.clustering.dirichlet.emitMostLikely";

  public static final String THRESHOLD_KEY = "org.apache.mahout.clustering.dirichlet.threshold";

  private static final Logger log = LoggerFactory.getLogger(DirichletDriver.class);

  private DirichletDriver() {
  }

  public static void main(String[] args) throws Exception {
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option maxIterOpt = DefaultOptionCreator.maxIterationsOption().create();
    Option kOpt = DefaultOptionCreator.kOption().withRequired(true).create();
    Option overwriteOutput = DefaultOptionCreator.overwriteOption().create();
    Option clusteringOpt = DefaultOptionCreator.clusteringOption().create();
    Option alphaOpt = DefaultOptionCreator.alphaOption().create();
    Option modelDistOpt = DefaultOptionCreator.modelDistributionOption().create();
    Option prototypeOpt = DefaultOptionCreator.modelPrototypeOption().create();
    Option numRedOpt = DefaultOptionCreator.numReducersOption().create();
    Option emitMostLikelyOpt = DefaultOptionCreator.emitMostLikelyOption().create();
    Option thresholdOpt = DefaultOptionCreator.thresholdOption().create();

    Group group = new GroupBuilder().withName("Options").withOption(inputOpt).withOption(outputOpt)
        .withOption(overwriteOutput).withOption(modelDistOpt).withOption(prototypeOpt)
        .withOption(maxIterOpt).withOption(alphaOpt).withOption(kOpt).withOption(helpOpt)
        .withOption(numRedOpt).withOption(clusteringOpt).withOption(emitMostLikelyOpt)
        .withOption(thresholdOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());
      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }
      String modelFactory = cmdLine.getValue(modelDistOpt).toString();
      String modelPrototype = cmdLine.getValue(prototypeOpt).toString();
      int numModels = Integer.parseInt(cmdLine.getValue(kOpt).toString());
      int numReducers = Integer.parseInt(cmdLine.getValue(numRedOpt).toString());
      int maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      boolean emitMostLikely = Boolean.parseBoolean(cmdLine.getValue(emitMostLikelyOpt).toString());
      double threshold = Double.parseDouble(cmdLine.getValue(thresholdOpt).toString());
      double alpha0 = Double.parseDouble(cmdLine.getValue(alphaOpt).toString());

      runJob(input, output, modelFactory, modelPrototype, numModels, maxIterations, alpha0, numReducers, cmdLine
          .hasOption(clusteringOpt), emitMostLikely, threshold);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   * @param modelFactory
   *          the String ModelDistribution class name to use
   * @param modelPrototype
   *          the String class name of the model prototype
   * @param numClusters
   *          the number of models
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param numReducers
   *          the number of Reducers desired
   * @param runClustering 
   *          true if clustering of points to be done after iterations
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   */
  public static void runJob(Path input,
                            Path output,
                            String modelFactory,
                            String modelPrototype,
                            int numClusters,
                            int maxIterations,
                            double alpha0,
                            int numReducers,
                            boolean runClustering,
                            boolean emitMostLikely,
                            double threshold)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException,
             SecurityException, NoSuchMethodException, InvocationTargetException {

    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);

    int protoSize = readPrototypeSize(input);

    writeInitialState(output, clustersIn, modelFactory, modelPrototype, protoSize, numClusters, alpha0);

    for (int iteration = 1; iteration <= maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      runIteration(input,
                   clustersIn,
                   clustersOut,
                   modelFactory,
                   modelPrototype,
                   protoSize,
                   numClusters,
                   alpha0,
                   numReducers);
      // now point the input to the old output directory
      clustersIn = clustersOut;
    }
    if (runClustering) {
      // now cluster the most likely points
      runClustering(input, clustersIn, new Path(output, Cluster.CLUSTERED_POINTS_DIR), emitMostLikely, threshold);
    }
  }

  private static int readPrototypeSize(Path input) throws IOException, InstantiationException, IllegalAccessException {
    JobConf job = new JobConf(DirichletDriver.class);
    FileSystem fs = FileSystem.get(input.toUri(), job);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int protoSize = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), job);
      WritableComparable key = (WritableComparable) reader.getKeyClass().newInstance();
      VectorWritable value = new VectorWritable();
      if (reader.next(key, value)) {
        protoSize = value.get().size();
      }
      reader.close();
      break;
    }
    return protoSize;
  }

  private static void writeInitialState(Path output,
                                        Path stateIn,
                                        String modelFactory,
                                        String modelPrototype,
                                        int prototypeSize,
                                        int numModels,
                                        double alpha0)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException, IOException,
             SecurityException, NoSuchMethodException, InvocationTargetException {

    DirichletState<VectorWritable> state = createState(modelFactory, modelPrototype, prototypeSize, numModels, alpha0);
    JobConf job = new JobConf(DirichletDriver.class);
    FileSystem fs = FileSystem.get(output.toUri(), job);
    for (int i = 0; i < numModels; i++) {
      Path path = new Path(stateIn, "part-" + i);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, Text.class, DirichletCluster.class);
      writer.append(new Text(Integer.toString(i)), state.getClusters().get(i));
      writer.close();
    }
  }

  /**
   * Creates a DirichletState object from the given arguments. Note that the modelFactory is presumed to be a
   * subclass of VectorModelDistribution that can be initialized with a concrete Vector prototype.
   * 
   * @param modelFactory
   *          a String which is the class name of the model factory
   * @param modelPrototype
   *          a String which is the class name of the Vector used to initialize the factory
   * @param prototypeSize
   *          an int number of dimensions of the model prototype vector
   * @param numModels
   *          an int number of models to be created
   * @param alpha0
   *          the double alpha_0 argument to the algorithm
   * @return an initialized DirichletState
   */
  public static DirichletState<VectorWritable> createState(String modelFactory,
                                                           String modelPrototype,
                                                           int prototypeSize,
                                                           int numModels,
                                                           double alpha0)
      throws ClassNotFoundException, InstantiationException, IllegalAccessException,
             SecurityException, NoSuchMethodException, IllegalArgumentException, InvocationTargetException {

    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<? extends VectorModelDistribution> cl = ccl.loadClass(modelFactory).asSubclass(VectorModelDistribution.class);
    VectorModelDistribution factory = cl.newInstance();

    Class<? extends Vector> vcl = ccl.loadClass(modelPrototype).asSubclass(Vector.class);
    Constructor<? extends Vector> v = vcl.getConstructor(int.class);
    factory.setModelPrototype(new VectorWritable(v.newInstance(prototypeSize)));
    return new DirichletState<VectorWritable>(factory, numModels, alpha0);
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param stateOut
   *          the directory pathname for output state
   * @param modelFactory
   *          the class name of the model factory class
   * @param modelPrototype
   *          the class name of the model prototype (a Vector implementation)
   * @param prototypeSize
   *          the size of the model prototype vector
   * @param numClusters
   *          the number of clusters
   * @param alpha0
   *          alpha_0
   * @param numReducers
   *          the number of Reducers desired
   */
  public static void runIteration(Path input,
                                  Path stateIn,
                                  Path stateOut,
                                  String modelFactory,
                                  String modelPrototype,
                                  int prototypeSize,
                                  int numClusters,
                                  double alpha0,
                                  int numReducers) {
    Configurable client = new JobClient();
    JobConf conf = new JobConf(DirichletDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(DirichletCluster.class);
    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(VectorWritable.class);

    FileInputFormat.setInputPaths(conf, input);
    FileOutputFormat.setOutputPath(conf, stateOut);

    conf.setMapperClass(DirichletMapper.class);
    conf.setReducerClass(DirichletReducer.class);
    conf.setNumReduceTasks(numReducers);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(MODEL_FACTORY_KEY, modelFactory);
    conf.set(MODEL_PROTOTYPE_KEY, modelPrototype);
    conf.set(PROTOTYPE_SIZE_KEY, Integer.toString(prototypeSize));
    conf.set(NUM_CLUSTERS_KEY, Integer.toString(numClusters));
    conf.set(ALPHA_0_KEY, Double.toString(alpha0));

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
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param output
   *          the directory pathname for output points
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   */
  public static void runClustering(Path input, Path stateIn, Path output, boolean emitMostLikely, double threshold) {
    JobConf conf = new JobConf(DirichletDriver.class);
    conf.setJobName("Dirichlet Clustering");

    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(conf, input);
    FileOutputFormat.setOutputPath(conf, output);

    conf.setMapperClass(DirichletClusterMapper.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // uncomment it to run locally
    // conf.set("mapred.job.tracker", "local");
    conf.setNumReduceTasks(0);
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(THRESHOLD_KEY, Double.toString(threshold));
    try {
      JobClient.runJob(conf);
    } catch (IOException e) {
      log.warn(e.toString(), e);
    }
  }
}
