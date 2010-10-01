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
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ModelDistribution;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.dirichlet.models.AbstractVectorModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.DistanceMeasureClusterDistribution;
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.clustering.kmeans.OutputLogFilter;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirichletDriver extends AbstractJob {

  public static final String STATE_IN_KEY = "org.apache.mahout.clustering.dirichlet.stateIn";

  public static final String MODEL_DISTRIBUTION_KEY = "org.apache.mahout.clustering.dirichlet.modelFactory";

  public static final String MODEL_PROTOTYPE_KEY = "org.apache.mahout.clustering.dirichlet.modelPrototype";

  public static final String PROTOTYPE_SIZE_KEY = "org.apache.mahout.clustering.dirichlet.prototypeSize";

  public static final String NUM_CLUSTERS_KEY = "org.apache.mahout.clustering.dirichlet.numClusters";

  public static final String ALPHA_0_KEY = "org.apache.mahout.clustering.dirichlet.alpha_0";

  public static final String EMIT_MOST_LIKELY_KEY = "org.apache.mahout.clustering.dirichlet.emitMostLikely";

  public static final String THRESHOLD_KEY = "org.apache.mahout.clustering.dirichlet.threshold";

  public static final String MODEL_PROTOTYPE_CLASS_OPTION = "modelPrototype";

  public static final String MODEL_DISTRIBUTION_CLASS_OPTION = "modelDist";

  public static final String ALPHA_OPTION = "alpha";

  private static final Logger log = LoggerFactory.getLogger(DirichletDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new DirichletDriver(), args);
  }

  @Override
  public int run(String[] args)
    throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException,
    NoSuchMethodException, InvocationTargetException, InterruptedException {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.numClustersOption().withRequired(true).create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(ALPHA_OPTION, "a0", "The alpha0 value for the DirichletDistribution. Defaults to 1.0", "1.0");
    addOption(MODEL_DISTRIBUTION_CLASS_OPTION,
              "md",
              "The ModelDistribution class name. Defaults to NormalModelDistribution",
              NormalModelDistribution.class.getName());
    addOption(MODEL_PROTOTYPE_CLASS_OPTION,
              "mp",
              "The ModelDistribution prototype Vector class name. Defaults to RandomAccessSparseVector",
              RandomAccessSparseVector.class.getName());
    addOption(DefaultOptionCreator.distanceMeasureOption().withRequired(false).create());
    addOption(DefaultOptionCreator.emitMostLikelyOption().create());
    addOption(DefaultOptionCreator.thresholdOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    String modelFactory = getOption(MODEL_DISTRIBUTION_CLASS_OPTION);
    String modelPrototype = getOption(MODEL_PROTOTYPE_CLASS_OPTION);
    String distanceMeasure = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    int numModels = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean emitMostLikely = Boolean.parseBoolean(getOption(DefaultOptionCreator.EMIT_MOST_LIKELY_OPTION));
    double threshold = Double.parseDouble(getOption(DefaultOptionCreator.THRESHOLD_OPTION));
    double alpha0 = Double.parseDouble(getOption(ALPHA_OPTION));
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential =
        getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD);
    int prototypeSize = readPrototypeSize(input);

    AbstractVectorModelDistribution modelDistribution = createModelDistribution(modelFactory,
                                                                                modelPrototype,
                                                                                distanceMeasure,
                                                                                prototypeSize);

    run(getConf(),
        input,
        output,
        modelDistribution,
        numModels,
        maxIterations,
        alpha0,
        runClustering,
        emitMostLikely,
        threshold,
        runSequential);
    return 0;
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * 
   * @param conf
   *          the Configuration to use
   * @param input
   *          the directory Path for input points
   * @param output
   *          the directory Path for output points
   * @param modelDistribution
   *          the String class name of the model's prototype vector
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param runClustering 
   *          true if clustering of points to be done after iterations
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential execute sequentially if true
   */
  public static void run(Configuration conf,
                         Path input,
                         Path output,
                         ModelDistribution<VectorWritable> modelDistribution,
                         int numModels,
                         int maxIterations,
                         double alpha0,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential)
    throws IOException, InstantiationException, ClassNotFoundException, InterruptedException, IllegalAccessException {
    Path clustersOut =
        buildClusters(conf, input, output, modelDistribution, numModels, maxIterations, alpha0, runSequential);
    if (runClustering) {
      clusterData(conf,
                  input,
                  clustersOut,
                  new Path(output, Cluster.CLUSTERED_POINTS_DIR),
                  emitMostLikely,
                  threshold,
                  runSequential);
    }
  }

  /**
   * Convenience method provides default Configuration
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * 
   * @param input
   *          the directory Path for input points
   * @param output
   *          the directory Path for output points
   * @param modelDistribution
   *          the String class name of the model's prototype vector
   * @param numClusters
   *          the number of models to iterate over
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param runClustering 
   *          true if clustering of points to be done after iterations
   * @param emitMostLikely
   *          a boolean if true emit only most likely cluster for each point
   * @param threshold 
   *          a double threshold value emits all clusters having greater pdf (emitMostLikely = false)
   * @param runSequential execute sequentially if true
   */
  public static void run(Path input,
                         Path output,
                         ModelDistribution<VectorWritable> modelDistribution,
                         int numClusters,
                         int maxIterations,
                         double alpha0,
                         boolean runClustering,
                         boolean emitMostLikely,
                         double threshold,
                         boolean runSequential)
    throws IOException, InstantiationException, IllegalAccessException, ClassNotFoundException, InterruptedException {
    run(new Configuration(),
        input,
        output,
        modelDistribution,
        numClusters,
        maxIterations,
        alpha0,
        runClustering,
        emitMostLikely,
        threshold,
        runSequential);
  }

  /**
   * Create an instance of AbstractVectorModelDistribution from the given command line arguments
   */
  public static AbstractVectorModelDistribution createModelDistribution(String modelFactory,
                                                                        String modelPrototype,
                                                                        String distanceMeasure,
                                                                        int prototypeSize)
    throws ClassNotFoundException, InstantiationException, IllegalAccessException,
    NoSuchMethodException, InvocationTargetException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<? extends AbstractVectorModelDistribution> cl = ccl.loadClass(modelFactory)
        .asSubclass(AbstractVectorModelDistribution.class);
    AbstractVectorModelDistribution modelDistribution = cl.newInstance();

    Class<? extends Vector> vcl = ccl.loadClass(modelPrototype).asSubclass(Vector.class);
    Constructor<? extends Vector> v = vcl.getConstructor(int.class);
    modelDistribution.setModelPrototype(new VectorWritable(v.newInstance(prototypeSize)));

    if (modelDistribution instanceof DistanceMeasureClusterDistribution) {
      Class<? extends DistanceMeasure> measureCl = ccl.loadClass(distanceMeasure).asSubclass(DistanceMeasure.class);
      DistanceMeasure measure = measureCl.newInstance();
      ((DistanceMeasureClusterDistribution) modelDistribution).setMeasure(measure);
    }
    return modelDistribution;
  }

  /**
   * Creates a DirichletState object from the given arguments. Note that the modelFactory is presumed to be a
   * subclass of VectorModelDistribution that can be initialized with a concrete Vector prototype.
   * 
   * @param modelDistribution the ModelDistribution
   * @param numModels an int number of models to be created
   * @param alpha0 the double alpha_0 argument to the algorithm
   * @return an initialized DirichletState
   */
  static DirichletState createState(ModelDistribution<VectorWritable> modelDistribution, int numModels, double alpha0) {
    return new DirichletState(modelDistribution, numModels, alpha0);
  }

  /**
   * Read the first input vector to determine the prototype size for the modelPrototype
   */
  public static int readPrototypeSize(Path input) throws IOException, InstantiationException, IllegalAccessException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int protoSize = 0;
    if (status.length > 0) {
      FileStatus s = status[0];
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
      VectorWritable value = new VectorWritable();
      if (reader.next(key, value)) {
        protoSize = value.get().size();
      }
      reader.close();
    }
    return protoSize;
  }

  /**
   * Write initial state (prior distribution) to the output path directory
   * @param output the output Path
   * @param stateOut the state output Path
   * @param modelDistribution the ModelDistribution
   * @param numModels the int number of models to generate
   * @param alpha0 the double alpha_0 argument to the DirichletDistribution
   */
  private static void writeInitialState(Path output,
                                        Path stateOut,
                                        ModelDistribution<VectorWritable> modelDistribution,
                                        int numModels,
                                        double alpha0) throws IOException {

    DirichletState state = createState(modelDistribution, numModels, alpha0);
    writeState(output, stateOut, numModels, state);
  }

  private static void writeState(Path output, Path stateOut, int numModels, DirichletState state) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    for (int i = 0; i < numModels; i++) {
      Path path = new Path(stateOut, "part-" + i);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, DirichletCluster.class);
      writer.append(new Text(Integer.toString(i)), state.getClusters().get(i));
      writer.close();
    }
  }

  /**
   * Run an iteration using supplied arguments
   * @param conf 
   * @param input the directory pathname for input points
   * @param stateIn the directory pathname for input state
   * @param stateOut the directory pathname for output state
   * @param modelDistribution the ModelDistribution
   * @param numClusters the number of clusters
   * @param alpha0 alpha_0
   */
  private static void runIteration(Configuration conf,
                                   Path input,
                                   Path stateIn,
                                   Path stateOut,
                                   ModelDistribution<VectorWritable> modelDistribution,
                                   int numClusters,
                                   double alpha0) throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(MODEL_DISTRIBUTION_KEY, modelDistribution.asJsonString());
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
    job.setJarByClass(DirichletDriver.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, stateOut);

    job.waitForCompletion(true);
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * @param conf 
   * @param input
   *          the directory Path for input points
   * @param output
   *          the directory Path for output points
   * @param modelDistribution
   *          the String class name of the model's prototype vector
   * @param numClusters
   *          the number of models to iterate over
   * @param maxIterations
   *          the maximum number of iterations
   * @param alpha0
   *          the alpha_0 value for the DirichletDistribution
   * @param runSequential execute sequentially if true
   * 
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf,
                                   Path input,
                                   Path output,
                                   ModelDistribution<VectorWritable> modelDistribution,
                                   int numClusters,
                                   int maxIterations,
                                   double alpha0,
                                   boolean runSequential)
    throws IOException, InstantiationException, ClassNotFoundException, InterruptedException, IllegalAccessException {
    Path clustersIn = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    writeInitialState(output, clustersIn, modelDistribution, numClusters, alpha0);

    if (runSequential) {
      clustersIn = buildClustersSeq(input, output, modelDistribution, numClusters, maxIterations, alpha0, clustersIn);
    } else {
      clustersIn = buildClustersMR(conf, input, output, modelDistribution, numClusters, maxIterations, alpha0, clustersIn);
    }
    return clustersIn;
  }

  private static Path buildClustersSeq(Path input,
                                       Path output,
                                       ModelDistribution<VectorWritable> modelDistribution,
                                       int numClusters,
                                       int maxIterations,
                                       double alpha0,
                                       Path clustersIn)
    throws IOException, InstantiationException, IllegalAccessException {
    for (int iteration = 1; iteration <= maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      DirichletState state = DirichletMapper.loadState(new Configuration(),
                                                       clustersIn.toString(),
                                                       modelDistribution,
                                                       alpha0,
                                                       numClusters);
      Cluster[] newModels = (Cluster[]) state.getModelFactory().sampleFromPosterior(state.getModels());
      DirichletClusterer clusterer = new DirichletClusterer(state);
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(input.toUri(), conf);
      FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
      for (FileStatus s : status) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
        try {
          Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
          VectorWritable vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
          while (reader.next(key, vw)) {
            clusterer.observe(newModels, vw);
            vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
          }
        } finally {
          reader.close();
        }
      }
      clusterer.updateModels(newModels);
      writeState(output, clustersOut, numClusters, state);

      // now point the input to the old output directory
      clustersIn = clustersOut;
    }
    return clustersIn;
  }

  private static Path buildClustersMR(Configuration conf,
                                      Path input,
                                      Path output,
                                      ModelDistribution<VectorWritable> modelDistribution,
                                      int numClusters,
                                      int maxIterations,
                                      double alpha0,
                                      Path clustersIn)
    throws IOException, InterruptedException, ClassNotFoundException {
    for (int iteration = 1; iteration <= maxIterations; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      runIteration(conf, input, clustersIn, clustersOut, modelDistribution, numClusters, alpha0);
      // now point the input to the old output directory
      clustersIn = clustersOut;
    }
    return clustersIn;
  }

  /**
   * Run the job using supplied arguments
   * @param conf 
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
   * @param runSequential execute sequentially if true
   */
  public static void clusterData(Configuration conf,
                                 Path input,
                                 Path stateIn,
                                 Path output,
                                 boolean emitMostLikely,
                                 double threshold,
                                 boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException, InstantiationException, IllegalAccessException {
    if (runSequential) {
      clusterDataSeq(input, stateIn, output, emitMostLikely, threshold);
    } else {
      clusterDataMR(conf, input, stateIn, output, emitMostLikely, threshold);
    }
  }

  private static void clusterDataSeq(Path input, Path stateIn, Path output, boolean emitMostLikely, double threshold)
      throws IOException, InstantiationException, IllegalAccessException {
    Configuration conf = new Configuration();
    List<DirichletCluster> clusters = DirichletClusterMapper.loadClusters(conf, stateIn);
    DirichletClusterer clusterer = new DirichletClusterer(emitMostLikely, threshold);
    // iterate over all points, assigning each to the closest canopy and outputing that clustering
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int part = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(output, "part-m-" + part++),
                                                           IntWritable.class,
                                                           WeightedVectorWritable.class);
      try {
        Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        VectorWritable vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        while (reader.next(key, vw)) {
          clusterer.emitPointToClusters(vw, clusters, writer);
          vw = reader.getValueClass().asSubclass(VectorWritable.class).newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }

  }

  private static void clusterDataMR(Configuration conf,
                                    Path input,
                                    Path stateIn,
                                    Path output,
                                    boolean emitMostLikely,
                                    double threshold) throws IOException, InterruptedException, ClassNotFoundException {
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
}
