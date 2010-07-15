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
import java.util.Map;

import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.dirichlet.models.AbstractVectorModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirichletDriver extends AbstractJob {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String MODEL_FACTORY_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String MODEL_PROTOTYPE_KEY = "org.apache.mahout.clustering.dirichlet.modelPrototype";

  public static final String PROTOTYPE_SIZE_KEY = "org.apache.mahout.clustering.dirichlet.prototypeSize";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";

  public static final String EMIT_MOST_LIKELY_KEY = "org.apache.mahout.clustering.dirichlet.emitMostLikely";

  public static final String THRESHOLD_KEY = "org.apache.mahout.clustering.dirichlet.threshold";

  protected static final String MODEL_PROTOTYPE_CLASS_OPTION = "modelPrototypeClass";

  public static final String MODEL_PROTOTYPE_CLASS_OPTION_KEY = "--" + MODEL_PROTOTYPE_CLASS_OPTION;

  protected static final String MODEL_DISTRIBUTION_CLASS_OPTION = "modelDistClass";

  public static final String MODEL_DISTRIBUTION_CLASS_OPTION_KEY = "--" + MODEL_DISTRIBUTION_CLASS_OPTION;

  protected static final String ALPHA_OPTION = "alpha";

  public static final String ALPHA_OPTION_KEY = "--" + ALPHA_OPTION;

  private static final Logger log = LoggerFactory.getLogger(DirichletDriver.class);

  protected DirichletDriver() {
  }

  public static void main(String[] args) throws Exception {
    new DirichletDriver().run(args);
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
   */
  public int run(String[] args) throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException,
      NoSuchMethodException, InvocationTargetException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(new DefaultOptionBuilder().withLongName(ALPHA_OPTION).withRequired(false).withShortName("m")
        .withArgument(new ArgumentBuilder().withName(ALPHA_OPTION).withDefault("1.0").withMinimum(1).withMaximum(1).create())
        .withDescription("The alpha0 value for the DirichletDistribution. Defaults to 1.0").create());
    addOption(new DefaultOptionBuilder().withLongName(MODEL_DISTRIBUTION_CLASS_OPTION).withRequired(false).withShortName("md")
        .withArgument(new ArgumentBuilder().withName(MODEL_DISTRIBUTION_CLASS_OPTION).withDefault(NormalModelDistribution.class
            .getName()).withMinimum(1).withMaximum(1).create()).withDescription("The ModelDistribution class name. "
            + "Defaults to NormalModelDistribution").create());
    addOption(new DefaultOptionBuilder().withLongName(MODEL_PROTOTYPE_CLASS_OPTION).withRequired(false).withShortName("mp")
        .withArgument(new ArgumentBuilder().withName("prototypeClass").withDefault(RandomAccessSparseVector.class.getName())
            .withMinimum(1).withMaximum(1).create())
        .withDescription("The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector").create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.numReducersOption().create());

    Map<String, String> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (argMap.containsKey(DefaultOptionCreator.OVERWRITE_OPTION_KEY)) {
      HadoopUtil.overwriteOutput(output);
    }
    String modelFactory = argMap.get(MODEL_DISTRIBUTION_CLASS_OPTION_KEY);
    String modelPrototype = argMap.get(MODEL_PROTOTYPE_CLASS_OPTION_KEY);
    int numModels = Integer.parseInt(argMap.get(DefaultOptionCreator.NUM_CLUSTERS_OPTION_KEY));
    int numReducers = Integer.parseInt(argMap.get(DefaultOptionCreator.MAX_REDUCERS_OPTION_KEY));
    int maxIterations = Integer.parseInt(argMap.get(DefaultOptionCreator.MAX_ITERATIONS_OPTION_KEY));
    boolean emitMostLikely = Boolean.parseBoolean(argMap.get(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION_KEY));
    double threshold = Double.parseDouble(argMap.get(DefaultOptionCreator.THRESHOLD_OPTION_KEY));
    double alpha0 = Double.parseDouble(argMap.get(ALPHA_OPTION_KEY));
    boolean runClustering = argMap.containsKey(DefaultOptionCreator.CLUSTERING_OPTION_KEY);

    job(input,
        output,
        modelFactory,
        modelPrototype,
        numModels,
        maxIterations,
        alpha0,
        numReducers,
        runClustering,
        emitMostLikely,
        threshold);
    return 0;
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
   * @throws InterruptedException 
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
                            double threshold) throws ClassNotFoundException, InstantiationException, IllegalAccessException,
      IOException, SecurityException, NoSuchMethodException, InvocationTargetException, InterruptedException {

    new DirichletDriver().job(input,
                              output,
                              modelFactory,
                              modelPrototype,
                              numClusters,
                              maxIterations,
                              alpha0,
                              numReducers,
                              runClustering,
                              emitMostLikely,
                              threshold);
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
  static DirichletState<VectorWritable> createState(String modelFactory,
                                                    String modelPrototype,
                                                    int prototypeSize,
                                                    int numModels,
                                                    double alpha0) throws ClassNotFoundException, InstantiationException,
      IllegalAccessException, SecurityException, NoSuchMethodException, IllegalArgumentException, InvocationTargetException {

    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<? extends AbstractVectorModelDistribution> cl = ccl.loadClass(modelFactory)
        .asSubclass(AbstractVectorModelDistribution.class);
    AbstractVectorModelDistribution factory = cl.newInstance();

    Class<? extends Vector> vcl = ccl.loadClass(modelPrototype).asSubclass(Vector.class);
    Constructor<? extends Vector> v = vcl.getConstructor(int.class);
    factory.setModelPrototype(new VectorWritable(v.newInstance(prototypeSize)));
    return new DirichletState<VectorWritable>(factory, numModels, alpha0);
  }

  /**
   * Read the first input vector to determine the prototype size for the modelPrototype
   * @param input
   * @return
   * @throws IOException
   * @throws InstantiationException
   * @throws IllegalAccessException
   */
  private int readPrototypeSize(Path input) throws IOException, InstantiationException, IllegalAccessException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int protoSize = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
      VectorWritable value = new VectorWritable();
      if (reader.next(key, value)) {
        protoSize = value.get().size();
      }
      reader.close();
      break;
    }
    return protoSize;
  }

  /**
   * Write initial state (prior distribution) to the output path directory
   * @param output the output Path
   * @param stateIn the state input Path
   * @param modelFactory the String class name of the modelFactory
   * @param modelPrototype the String class name of the modelPrototype
   * @param prototypeSize the int size of the modelPrototype vectors
   * @param numModels the int number of models to generate
   * @param alpha0 the double alpha_0 argument to the DirichletDistribution
   * @throws ClassNotFoundException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws IOException
   * @throws SecurityException
   * @throws NoSuchMethodException
   * @throws InvocationTargetException
   */
  private void writeInitialState(Path output,
                                 Path stateIn,
                                 String modelFactory,
                                 String modelPrototype,
                                 int prototypeSize,
                                 int numModels,
                                 double alpha0) throws ClassNotFoundException, InstantiationException, IllegalAccessException,
      IOException, SecurityException, NoSuchMethodException, InvocationTargetException {

    DirichletState<VectorWritable> state = createState(modelFactory, modelPrototype, prototypeSize, numModels, alpha0);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    for (int i = 0; i < numModels; i++) {
      Path path = new Path(stateIn, "part-" + i);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, DirichletCluster.class);
      writer.append(new Text(Integer.toString(i)), state.getClusters().get(i));
      writer.close();
    }
  }

  /**
   * Run an iteration using supplied arguments
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
   * @throws IOException 
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private void runIteration(Path input,
                            Path stateIn,
                            Path stateOut,
                            String modelFactory,
                            String modelPrototype,
                            int prototypeSize,
                            int numClusters,
                            double alpha0,
                            int numReducers) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(MODEL_FACTORY_KEY, modelFactory);
    conf.set(MODEL_PROTOTYPE_KEY, modelPrototype);
    conf.set(PROTOTYPE_SIZE_KEY, Integer.toString(prototypeSize));
    conf.set(NUM_CLUSTERS_KEY, Integer.toString(numClusters));
    conf.set(ALPHA_0_KEY, Double.toString(alpha0));

    Job job = new Job(conf);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(DirichletCluster.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setMapperClass(DirichletMapper.class);
    job.setReducerClass(DirichletReducer.class);
    job.setNumReduceTasks(numReducers);
    job.setJarByClass(DirichletDriver.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, stateOut);

    job.waitForCompletion(true);
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
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IOException 
   */
  private void runClustering(Path input, Path stateIn, Path output, boolean emitMostLikely, double threshold) throws IOException,
      InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(EMIT_MOST_LIKELY_KEY, Boolean.toString(emitMostLikely));
    conf.set(THRESHOLD_KEY, Double.toString(threshold));
    Job job = new Job(conf);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);
    job.setMapperClass(DirichletClusterMapper.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(DirichletDriver.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.waitForCompletion(true);
  }

  /**
   * Run the job
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
   * @throws IOException
   * @throws InstantiationException
   * @throws IllegalAccessException
   * @throws ClassNotFoundException
   * @throws NoSuchMethodException
   * @throws InvocationTargetException
   * @throws InterruptedException
   */
  private void job(Path input,
                   Path output,
                   String modelFactory,
                   String modelPrototype,
                   int numClusters,
                   int maxIterations,
                   double alpha0,
                   int numReducers,
                   boolean runClustering,
                   boolean emitMostLikely,
                   double threshold) throws IOException, InstantiationException, IllegalAccessException, ClassNotFoundException,
      NoSuchMethodException, InvocationTargetException, InterruptedException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);

    int protoSize = readPrototypeSize(input);

    writeInitialState(output, clustersIn, modelFactory, modelPrototype, protoSize, numClusters, alpha0);

    for (int iteration = 1; iteration <= maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      runIteration(input, clustersIn, clustersOut, modelFactory, modelPrototype, protoSize, numClusters, alpha0, numReducers);
      // now point the input to the old output directory
      clustersIn = clustersOut;
    }
    if (runClustering) {
      // now cluster the most likely points
      runClustering(input, clustersIn, new Path(output, Cluster.CLUSTERED_POINTS_DIR), emitMostLikely, threshold);
    }
  }
}
