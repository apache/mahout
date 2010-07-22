/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.kmeans;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KMeansDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  public KMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    new KMeansDriver().run(args);
  }

  /**
   * Run the job using supplied arguments on a new driver instance (convenience)
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @param runSequential if true execute sequential algorithm 
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public static void runJob(Path input,
                            Path clustersIn,
                            Path output,
                            String measureClass,
                            double convergenceDelta,
                            int maxIterations,
                            int numReduceTasks,
                            boolean runClustering,
                            boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    new KMeansDriver().job(input,
                           clustersIn,
                           output,
                           measureClass,
                           convergenceDelta,
                           maxIterations,
                           numReduceTasks,
                           runClustering,
                           runSequential);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.clustersInOption()
        .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected" + " and written out to this path first")
        .create());
    addOption(DefaultOptionCreator.numClustersOption()
        .withDescription("The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.numReducersOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path clusters = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int numReduceTasks = Integer.parseInt(getOption(DefaultOptionCreator.MAX_REDUCERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(input, clusters, Integer
          .parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)));
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = (getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(DefaultOptionCreator.SEQUENTIAL_METHOD));
    job(input, clusters, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, runClustering, runSequential);
    return 0;
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @param runSequential if true execute sequential algorithm
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void job(Path input,
                  Path clustersIn,
                  Path output,
                  String measureClass,
                  double convergenceDelta,
                  int maxIterations,
                  int numReduceTasks,
                  boolean runClustering,
                  boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException, InstantiationException,
      IllegalAccessException {
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    Class<?> cl = ccl.loadClass(measureClass);
    DistanceMeasure measure = (DistanceMeasure) cl.newInstance();

    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}", new Object[] { input, clustersIn, output, measureClass });
      log.info("convergence: {} max Iterations: {} num Reduce Tasks: {} Input Vectors: {}", new Object[] { convergenceDelta,
          maxIterations, numReduceTasks, VectorWritable.class.getName() });
    }
    Path clustersOut = buildClusters(input, clustersIn, output, measure, maxIterations, numReduceTasks, delta, runSequential);
    if (runClustering) {
      log.info("Clustering data");
      clusterData(input, clustersOut, new Path(output, Cluster.CLUSTERED_POINTS_DIR), measure, delta, runSequential);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runSequential if true execute sequential algorithm
   * @param convergenceDelta
   *          the convergence delta value
   * @return the Path of the final clusters directory
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public Path buildClusters(Path input,
                            Path clustersIn,
                            Path output,
                            DistanceMeasure measure,
                            int maxIterations,
                            int numReduceTasks,
                            String delta,
                            boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    if (runSequential) {
      return buildClustersSeq(input, clustersIn, output, measure, maxIterations, numReduceTasks, delta);
    } else {
      return buildClustersMR(input, clustersIn, output, measure, maxIterations, numReduceTasks, delta);
    }
  }

  /**
   * @param input
   * @param clustersIn
   * @param output
   * @param measure
   * @param maxIterations
   * @param numReduceTasks
   * @param delta
   * @return
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  private Path buildClustersSeq(Path input,
                                Path clustersIn,
                                Path output,
                                DistanceMeasure measure,
                                int maxIterations,
                                int numReduceTasks,
                                String delta) throws InstantiationException, IllegalAccessException, IOException,
      ClassNotFoundException {
    KMeansClusterer clusterer = new KMeansClusterer(measure);
    List<Cluster> clusters = new ArrayList<Cluster>();

    KMeansUtil.configureWithClusterInfo(clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      Configuration conf = new Configuration();
      FileSystem fs = FileSystem.get(input.toUri(), conf);
      FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
      for (FileStatus s : status) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
        try {
          WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
          VectorWritable vw = (VectorWritable) reader.getValueClass().newInstance();
          while (reader.next(key, vw)) {
            clusterer.addPointToNearestCluster(vw.get(), clusters);
            vw = (VectorWritable) reader.getValueClass().newInstance();
          }
        } finally {
          reader.close();
        }
      }
      converged = clusterer.testConvergence(clusters, Double.parseDouble(delta));
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(clustersOut, "part-r-00000"),
                                                           Text.class,
                                                           Cluster.class);
      try {
        for (Cluster cluster : clusters) {
          writer.append(new Text(cluster.getIdentifier()), cluster);
        }
      } finally {
        writer.close();
      }
      clustersIn = clustersOut;
    }
    return clustersIn;
  }

  /**
   * @param input
   * @param clustersIn
   * @param output
   * @param measure
   * @param maxIterations
   * @param numReduceTasks
   * @param delta
   * @return
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private Path buildClustersMR(Path input,
                               Path clustersIn,
                               Path output,
                               DistanceMeasure measure,
                               int maxIterations,
                               int numReduceTasks,
                               String delta) throws IOException, InterruptedException, ClassNotFoundException {
    boolean converged = false;
    int iteration = 1;
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input, clustersIn, clustersOut, measure.getClass().getName(), delta, numReduceTasks);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }
    return clustersIn;
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param clustersOut
   *          the directory pathname for output clusters
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param numReduceTasks
   *          the number of reducer tasks
   * @return true if the iteration successfully runs
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private boolean runIteration(Path input,
                               Path clustersIn,
                               Path clustersOut,
                               String measureClass,
                               String convergenceDelta,
                               int numReduceTasks) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(KMeansInfo.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Cluster.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(KMeansMapper.class);
    job.setCombinerClass(KMeansCombiner.class);
    job.setReducerClass(KMeansReducer.class);
    job.setNumReduceTasks(numReduceTasks);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    job.setJarByClass(KMeansDriver.class);
    HadoopUtil.overwriteOutput(clustersOut);
    job.waitForCompletion(true);
    FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);

    return isConverged(clustersOut, conf, fs);
  }

  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @param conf
   *          the JobConf
   * @param fs
   *          the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   */
  private boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {
    FileStatus[] parts = fs.listStatus(filePath);
    for (FileStatus part : parts) {
      String name = part.getPath().getName();
      if (name.startsWith("part") && !name.endsWith(".crc")) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, part.getPath(), conf);
        try {
          Writable key = (Writable) reader.getKeyClass().newInstance();
          Cluster value = new Cluster();
          while (reader.next(key, value)) {
            if (!value.isConverged()) {
              return false;
            }
          }
        } catch (InstantiationException e) { // shouldn't happen
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } catch (IllegalAccessException e) {
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } finally {
          reader.close();
        }
      }
    }
    return true;
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output points
   * @param measure
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param runSequential if true execute sequential algorithm
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  public void clusterData(Path input,
                          Path clustersIn,
                          Path output,
                          DistanceMeasure measure,
                          String convergenceDelta,
                          boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException,
      InstantiationException, IllegalAccessException {
    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}", new Object[] { input, clustersIn, output, measure });
      log.info("convergence: {} Input Vectors: {}", convergenceDelta, VectorWritable.class.getName());
    }
    if (runSequential) {
      clusterDataSeq(input, clustersIn, output, measure, convergenceDelta);
    } else {
      clusterDataMR(input, clustersIn, output, measure, convergenceDelta);
    }
  }

  /**
   * @param input
   * @param clustersIn
   * @param output
   * @param measure
   * @param convergenceDelta
   * @throws InterruptedException 
   * @throws IOException 
   * @throws IllegalAccessException 
   * @throws InstantiationException 
   */
  private void clusterDataSeq(Path input, Path clustersIn, Path output, DistanceMeasure measure, String convergenceDelta)
      throws IOException, InterruptedException, InstantiationException, IllegalAccessException {
    KMeansClusterer clusterer = new KMeansClusterer(measure);
    List<Cluster> clusters = new ArrayList<Cluster>();
    KMeansUtil.configureWithClusterInfo(clustersIn, clusters);
    if (clusters.isEmpty()) {
      throw new IllegalStateException("Clusters is empty!");
    }
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    FileStatus[] status = fs.listStatus(input, new OutputLogFilter());
    int part = 0;
    for (FileStatus s : status) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, s.getPath(), conf);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs,
                                                           conf,
                                                           new Path(output, "part-m-" + part),
                                                           IntWritable.class,
                                                           WeightedVectorWritable.class);
      try {
        WritableComparable<?> key = (WritableComparable<?>) reader.getKeyClass().newInstance();
        VectorWritable vw = (VectorWritable) reader.getValueClass().newInstance();
        while (reader.next(key, vw)) {
          clusterer.emitPointToNearestCluster(vw.get(), clusters, writer);
          vw = (VectorWritable) reader.getValueClass().newInstance();
        }
      } finally {
        reader.close();
        writer.close();
      }
    }

  }

  /**
   * @param input
   * @param clustersIn
   * @param output
   * @param measure
   * @param convergenceDelta
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  private void clusterDataMR(Path input, Path clustersIn, Path output, DistanceMeasure measure, String convergenceDelta)
      throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measure.getClass().getName());
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    HadoopUtil.overwriteOutput(output);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(KMeansClusterMapper.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(KMeansDriver.class);

    job.waitForCompletion(true);
  }
}
